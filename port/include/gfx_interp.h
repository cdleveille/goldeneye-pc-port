#ifndef PORT_GFX_INTERP_H
#define PORT_GFX_INTERP_H

/* D573: frame interpolation (Video.Interpolation). Port-only.
 *
 * The game simulates and builds one display list per 60 Hz tick (50 Hz PAL).
 * With interpolation on, the render worker (D481, port/src/libultra.c) draws
 * each display list several times, at the display's rate, and fast3d blends
 * every vertex batch's transform between the previous tick and this one:
 *
 *   - Every batch is keyed so the same geometry can be found in the previous
 *     tick (static vertex source + occurrence, or model instance + ordinal).
 *   - The camera is blended separately (from the player's own world-to-camera
 *     matrix), so world geometry moves with an exact slerped camera and the
 *     floating room origin (bgroomtrans.c) cancels out.
 *   - Unmatched world geometry (effects built fresh each tick) follows the
 *     blended camera at its newest position; the first-person gun and hands
 *     (drawn under their own G_MW_PERSPNORM value) and z-buffer-off geometry
 *     are camera-anchored and blended in camera space.
 *   - The CPU-projected sky (G_FLOATVTX_EXT) is re-projected by the camera's
 *     rotation change, and the rooms' portal clip rectangles (G_SETSCISSOR)
 *     are moved with the displayed view.
 *   - Optional mouse-look prediction turns the view by mouse motion the game
 *     has not consumed yet, so mouse look runs at the display rate.
 *
 * Game logic and timing are untouched: the worker posts the same done events
 * after the first draw of each display list, exactly as with interpolation
 * off. Off by default; when off none of this code runs. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GFX_INTERP_MAX_CAMS 4

typedef struct GfxInterpCam {
    int valid;
    const void *proj;   /* the scene projection Mtx the DL loads (player->projmatrix) */
    const void *pv;     /* projection x view Mtx for level geometry (player->field_10E0) */
    float V[4][4];      /* world -> camera, row vectors (player->field_10CC,
                           times the stage's visibility scale) */
    float P[4][4];      /* proj, decoded */
} GfxInterpCam;

typedef struct GfxInterpFrame {
    int32_t sim_tick;          /* currentFrameCounter (game ticks) for this DL */
    int32_t stage;             /* stage being shown; no blending across a change */
    uint64_t submit_us;        /* host time the DL was submitted */
    uintptr_t dyn_lo[2];       /* per-tick memory pools (dyn.c), for "is this */
    uintptr_t dyn_hi[2];       /* vertex source rebuilt every tick?"          */
    GfxInterpCam cams[GFX_INTERP_MAX_CAMS];
    int look_cam;              /* camera the mouse turns (-1 = none) */
    double look_total_yaw;     /* input.c running mouse-look totals at submit */
    double look_total_pitch;
    float look_yaw_deg;        /* mouse look applied between the previous DL and */
    float look_pitch_deg;      /* this one (set by the render worker)            */
} GfxInterpFrame;

typedef struct GfxInterpPass {
    float alpha;               /* 0 = previous tick, 1 = this tick */
    int predict;               /* apply the pending mouse look below */
    float pend_yaw_deg;        /* mouse look not in this DL yet: applied since */
    float pend_pitch_deg;      /* it was submitted, plus motion no poll read   */
} GfxInterpPass;

/* Render thread only. Before the first gfx_run of a new display list: */
void gfx_interp_new_frame(const GfxInterpFrame *f, const GfxInterpPass *pass);
/* Before a repeat gfx_run of the same display list: */
void gfx_interp_replay(const GfxInterpPass *pass);
/* Interpolation off for the next gfx_run; forget history. */
void gfx_interp_disable(void);
/* After a gfx_run: 1 if the display list can be drawn again. */
int gfx_interp_can_replay(void);
/* Wall time (ms) the last recorded pass spent walking the display list. */
double gfx_interp_last_dl_ms(void);
/* GE_INTERP_LOG level (0 off, 1 per second, 2 also per draw). */
int gfx_interp_log_level(void);

#ifdef __cplusplus
}
#endif

#endif
