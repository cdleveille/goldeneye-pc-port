/* D573: frame interpolation -- the game-state snapshot taken when a display
 * list is submitted (port/src/libultra.c osSpTaskStartGo, scheduler thread).
 *
 * Read-only: per player, the scene projection and projection x view matrices
 * the display list loads (by pointer, so fast3d can recognise them), the
 * world-to-camera matrix (bondview2.c bondviewUpdateCameraMatrices, with the
 * stage's visibility scale folded in), and
 * input.c's running total of the mouse look it wrote into the player. The
 * game is between frames here: it submits the task, then waits for the next
 * VI retrace before it touches any of this again. */

#include "port_math.h"   /* real system math decls; see header */
#include <string.h>

#include <ultra64.h>
#include "bondtypes.h"
#include "bondconstants.h"
#include "bondview.h"
#include "player.h"

#include "system.h"
#include "input.h"
#include "gfx_interp.h"

extern s32 currentFrameCounter;   /* src/game/frametiming.c */
extern u8 *g_GfxBuffers[3];       /* src/game/dyn.c */
extern u8 *g_VtxBuffers[3];
extern s32 getPlayerCount(void);
extern s32 lvlGetCurrentStageToLoad(void);

/* Fixed-point Mtx -> float, as fast3d's gfx_sp_matrix decodes it. */
static void interpDecodeMtx(const void *mtx, float m[4][4])
{
    const int32_t *a = (const int32_t *)mtx;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j += 2) {
            int32_t ip = a[i * 2 + j / 2];
            uint32_t fp = (uint32_t)a[8 + i * 2 + j / 2];
            m[i][j] = (int32_t)(((uint32_t)ip & 0xffff0000u) | (fp >> 16)) / 65536.0f;
            m[i][j + 1] = (int32_t)(((uint32_t)ip << 16) | (fp & 0xffffu)) / 65536.0f;
        }
    }
}

static int interpInRange(const void *p, uintptr_t lo, uintptr_t hi)
{
    return (uintptr_t)p >= lo && (uintptr_t)p + 64 <= hi;
}

/* A world-to-camera matrix: rigid, unit-length rotation rows. */
static int interpViewOk(float V[4][4])
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (!isfinite(V[i][j])) {
                return 0;
            }
        }
    }
    if (fabsf(V[0][3]) > 1e-3f || fabsf(V[1][3]) > 1e-3f || fabsf(V[2][3]) > 1e-3f || fabsf(V[3][3] - 1.0f) > 1e-3f) {
        return 0;
    }
    for (int i = 0; i < 3; i++) {
        float n = V[i][0] * V[i][0] + V[i][1] * V[i][1] + V[i][2] * V[i][2];
        if (fabsf(n - 1.0f) > 0.02f) {
            return 0;
        }
    }
    return 1;
}

void portInterpSnapshot(GfxInterpFrame *f, const void *dl)
{
    memset(f, 0, sizeof(*f));
    f->sim_tick = currentFrameCounter;
    f->stage = lvlGetCurrentStageToLoad();
    f->submit_us = sysGetMicroseconds();
    f->look_cam = -1;
    f->dyn_lo[0] = (uintptr_t)g_GfxBuffers[0];
    f->dyn_hi[0] = (uintptr_t)g_GfxBuffers[2];
    f->dyn_lo[1] = (uintptr_t)g_VtxBuffers[0];
    f->dyn_hi[1] = (uintptr_t)g_VtxBuffers[2];

    inputLookTotals(&f->look_total_yaw, &f->look_total_pitch);

    /* The half of the per-tick pools this display list was built in: the
     * camera matrices must have been allocated for it, not left over. */
    int half = -1;
    for (int i = 0; i < 2; i++) {
        if ((uintptr_t)dl >= (uintptr_t)g_GfxBuffers[i] && (uintptr_t)dl < (uintptr_t)g_GfxBuffers[i + 1]) {
            half = i;
        }
    }
    if (half < 0 || f->stage == LEVELID_TITLE || f->stage <= 0) {
        return;   /* front end / no level: matching only, no scene cameras */
    }
    const uintptr_t lo = (uintptr_t)g_VtxBuffers[half];
    const uintptr_t hi = (uintptr_t)g_VtxBuffers[half + 1];

    const s32 n = getPlayerCount();
    for (s32 k = 0; k < n && k < GFX_INTERP_MAX_CAMS; k++) {
        struct player *p = g_playerPointers[k];
        if (p == NULL) {
            continue;
        }
        if (!interpInRange(p->projmatrix, lo, hi) || !interpInRange(p->field_10E0, lo, hi) ||
            !interpInRange(p->field_10CC, lo, hi)) {
            continue;
        }
        GfxInterpCam *c = &f->cams[k];
        memcpy(c->V, p->field_10CC, sizeof(c->V));
        if (!interpViewOk(c->V)) {
            continue;
        }
        /* Rooms and models are drawn in a camera space scaled by the stage's
         * visibility scale (bg.c levelinfotable: 0.2 on Dam and Surface): the
         * room and model matrices go through matrix_4x4_f32_to_s32 with it,
         * and the level view (field_10E0) is built from the scaled camera
         * offset (bondview2.c). field_10CC is unscaled; fold the scale in so
         * the world transforms fast3d derives from it are in world units. */
        if (isfinite(D_800364CC) && D_800364CC > 0.0f && D_800364CC != 1.0f) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    c->V[i][j] *= D_800364CC;
                }
            }
        }
        interpDecodeMtx(p->projmatrix, c->P);
        c->proj = p->projmatrix;
        c->pv = p->field_10E0;
        c->valid = 1;
    }
    if (n == 1 && f->cams[0].valid) {
        f->look_cam = 0;
    }
    if (n >= 1 && f->cams[0].valid && gfx_interp_log_level() >= 2) {
        /* GE_INTERP_LOG=2: the camera this list was built with vs the yaw the
         * game holds now and the mouse look applied so far */
        const struct player *p = g_playerPointers[0];
        const float h = atan2f(f->cams[0].V[0][2], -f->cams[0].V[2][2]) * (180.0f / 3.14159265f);
        sysLogPrintf(LOG_NOTE, "D573 snap sim=%d theta=%.4f cam_h=%.4f total=%.4f",
                     (int)f->sim_tick, (double)p->vv_theta, (double)h, f->look_total_yaw);
    }
}
