/* D573: frame interpolation core. See port/include/gfx_interp.h for the
 * design; the hooks are called from gfx_pc.cpp on the render thread.
 *
 * Conventions: fast3d's row-vector matrices (v' = v * M; M[3] is the
 * translation row), MP = MV * P. "C" is a batch's camera-space transform,
 * "W" its world transform relative to its own tick's camera position (C times
 * the inverse of V's rotation and scale), V the camera's world-to-camera
 * matrix. W is kept camera-relative because C is only exact up to a
 * homogeneous scale (level geometry's C comes from the fixed-point projection
 * x view, divided by the fixed-point projection): harmless around the camera,
 * but tens of units of error when multiplied out to absolute world
 * coordinates (~20000 on Dam).
 *
 * Each display list is recorded on its first draw: every G_MTX value and its
 * classification, and one record per vertex batch with its transforms and the
 * batch it matches in the previous display list. Replays walk fast3d's
 * flattened copy of that first walk (gfx_pc.cpp, s_rec_gfx -- the game is
 * already rewriting its own display lists by then) and look the records up by
 * ordinal. */

#define NOMINMAX
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <vector>

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif
#include <PR/gbi.h>

#include "system.h"
#include "portaddr.h"
#include "gfx_interp.h"
#include "gfx_interp_hooks.h"

namespace {

/* Emulated DRAM is mapped twice (port/src/dram.c): display lists reach the
 * same bytes through the V1 view or the KSEG0 mirror depending on how the
 * game wrote the address. Compare in the V1 view. */
inline uintptr_t norm(uintptr_t p) {
    return (p >= PORT_DRAM_K0_BASE && p < PORT_DRAM_K0_BASE + PORT_DRAM_SIZE)
               ? p - PORT_DRAM_K0_BASE + PORT_DRAM_V1_BASE : p;
}

typedef float M4[4][4];

/* Tunables, in GE world units per game tick. */
const float kCamCutDist = 400.0f;       /* camera moved further in one tick: a cut */
const float kCamCutAngle = 1.22f;       /* ~70 degrees of view rotation in one tick: a cut */
const float kObjSnapDist = 400.0f;      /* object moved further in one tick: don't blend */
const float kObjSnapAngle = 2.0f;       /* ~115 degrees in one tick */
const float kInstMatchDist = 300.0f;    /* max anchor distance when pairing model instances */
const float kScissorDepth = 100.0f;     /* nearest portal a moved room clip rectangle allows for */
const uintptr_t kInstMtxSpan = 0x40 * 256;  /* direct pointers into a model's matrix array */
const int kMaxTicksBlend = 8;           /* more ticks between two display lists: don't blend */

/* ---------------------------------------------------------------- math --- */

void m4_copy(M4 d, const M4 s) { memcpy(d, s, sizeof(M4)); }

void m4_mul(M4 out, const M4 a, const M4 b) {
    M4 t;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            t[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    m4_copy(out, t);
}

void m4_lerp(M4 out, const M4 a, const M4 b, float t) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i][j] = a[i][j] + (b[i][j] - a[i][j]) * t;
        }
    }
}

bool m4_invert(M4 out, const M4 in) {
    double m[16], inv[16];
    for (int i = 0; i < 16; i++) {
        m[i] = in[i / 4][i % 4];
    }
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (!(fabs(det) > 1e-12)) {
        return false;
    }
    det = 1.0 / det;
    for (int i = 0; i < 16; i++) {
        out[i / 4][i % 4] = (float)(inv[i] * det);
    }
    return true;
}

void m3_mul(float out[3][3], const float a[3][3], const float b[3][3]) {
    float t[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            t[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    memcpy(out, t, sizeof(t));
}

/* Quaternion <-> 3x3, one consistent pair (which handedness the 3x3 uses
 * does not matter: the round trip is exact and slerp commutes with it). */
void quat_from_m3(const float r[3][3], float q[4]) {
    const float tr = r[0][0] + r[1][1] + r[2][2];
    if (tr > 0.0f) {
        const float s = sqrtf(tr + 1.0f) * 2.0f;
        q[3] = 0.25f * s;
        q[0] = (r[2][1] - r[1][2]) / s;
        q[1] = (r[0][2] - r[2][0]) / s;
        q[2] = (r[1][0] - r[0][1]) / s;
    } else if (r[0][0] > r[1][1] && r[0][0] > r[2][2]) {
        const float s = sqrtf(1.0f + r[0][0] - r[1][1] - r[2][2]) * 2.0f;
        q[3] = (r[2][1] - r[1][2]) / s;
        q[0] = 0.25f * s;
        q[1] = (r[0][1] + r[1][0]) / s;
        q[2] = (r[0][2] + r[2][0]) / s;
    } else if (r[1][1] > r[2][2]) {
        const float s = sqrtf(1.0f + r[1][1] - r[0][0] - r[2][2]) * 2.0f;
        q[3] = (r[0][2] - r[2][0]) / s;
        q[0] = (r[0][1] + r[1][0]) / s;
        q[1] = 0.25f * s;
        q[2] = (r[1][2] + r[2][1]) / s;
    } else {
        const float s = sqrtf(1.0f + r[2][2] - r[0][0] - r[1][1]) * 2.0f;
        q[3] = (r[1][0] - r[0][1]) / s;
        q[0] = (r[0][2] + r[2][0]) / s;
        q[1] = (r[1][2] + r[2][1]) / s;
        q[2] = 0.25f * s;
    }
    const float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 0.0f) {
        for (int i = 0; i < 4; i++) {
            q[i] /= n;
        }
    }
}

void m3_from_quat(const float q[4], float r[3][3]) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    r[0][0] = 1.0f - 2.0f * (y * y + z * z);
    r[0][1] = 2.0f * (x * y - z * w);
    r[0][2] = 2.0f * (x * z + y * w);
    r[1][0] = 2.0f * (x * y + z * w);
    r[1][1] = 1.0f - 2.0f * (x * x + z * z);
    r[1][2] = 2.0f * (y * z - x * w);
    r[2][0] = 2.0f * (x * z - y * w);
    r[2][1] = 2.0f * (y * z + x * w);
    r[2][2] = 1.0f - 2.0f * (x * x + y * y);
}

void q_slerp(float out[4], const float a[4], const float b_in[4], float t) {
    float b[4] = { b_in[0], b_in[1], b_in[2], b_in[3] };
    float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if (d < 0.0f) {
        for (int i = 0; i < 4; i++) {
            b[i] = -b[i];
        }
        d = -d;
    }
    float wa, wb;
    if (d > 0.9995f) {
        wa = 1.0f - t;
        wb = t;
    } else {
        const float th = acosf(d);
        const float s = sinf(th);
        wa = sinf((1.0f - t) * th) / s;
        wb = sinf(t * th) / s;
    }
    float n = 0.0f;
    for (int i = 0; i < 4; i++) {
        out[i] = a[i] * wa + b[i] * wb;
        n += out[i] * out[i];
    }
    n = sqrtf(n);
    if (n > 0.0f) {
        for (int i = 0; i < 4; i++) {
            out[i] /= n;
        }
    }
}

float q_angle(const float a[4], const float b[4]) {
    float d = fabsf(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
    if (d > 1.0f) {
        d = 1.0f;
    }
    return 2.0f * acosf(d);
}

/* An affine transform split into rotation, per-axis scale (rows) and
 * translation. Fails for projective matrices, degenerate rows and shear. */
struct Aff {
    float q[4];
    float s[3];
    float t[3];
};

bool aff_decompose(const M4 m, Aff &a) {
    if (fabsf(m[0][3]) > 1e-3f || fabsf(m[1][3]) > 1e-3f || fabsf(m[2][3]) > 1e-3f || fabsf(m[3][3] - 1.0f) > 1e-3f) {
        return false;
    }
    float r[3][3];
    for (int i = 0; i < 3; i++) {
        const float s = sqrtf(m[i][0] * m[i][0] + m[i][1] * m[i][1] + m[i][2] * m[i][2]);
        if (!(s > 1e-6f)) {
            return false;
        }
        a.s[i] = s;
        for (int j = 0; j < 3; j++) {
            r[i][j] = m[i][j] / s;
        }
    }
    const float det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
                      r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
                      r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if (det < 0.0f) {
        a.s[0] = -a.s[0];
        for (int j = 0; j < 3; j++) {
            r[0][j] = -r[0][j];
        }
    }
    const float d01 = r[0][0] * r[1][0] + r[0][1] * r[1][1] + r[0][2] * r[1][2];
    const float d02 = r[0][0] * r[2][0] + r[0][1] * r[2][1] + r[0][2] * r[2][2];
    const float d12 = r[1][0] * r[2][0] + r[1][1] * r[2][1] + r[1][2] * r[2][2];
    if (fabsf(d01) > 0.03f || fabsf(d02) > 0.03f || fabsf(d12) > 0.03f) {
        return false;
    }
    quat_from_m3(r, a.q);
    for (int j = 0; j < 3; j++) {
        a.t[j] = m[3][j];
    }
    return true;
}

void aff_compose(M4 m, const Aff &a) {
    float r[3][3];
    m3_from_quat(a.q, r);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m[i][j] = r[i][j] * a.s[i];
        }
        m[i][3] = 0.0f;
        m[3][i] = a.t[i];
    }
    m[3][3] = 1.0f;
}

void aff_blend(Aff &out, const Aff &a, const Aff &b, float t) {
    q_slerp(out.q, a.q, b.q, t);
    for (int i = 0; i < 3; i++) {
        out.s[i] = a.s[i] + (b.s[i] - a.s[i]) * t;
        out.t[i] = a.t[i] + (b.t[i] - a.t[i]) * t;
    }
}

float dist3(const float a[3], const float b[3]) {
    const float x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return sqrtf(x * x + y * y + z * z);
}

/* Rotate a camera (R = the 3x3 of a world-to-camera matrix: its columns are
 * the camera axes in world space) by the game's own yaw/pitch: GE's forward
 * is (-sin yaw, sin pitch, cos yaw) (bondview2.c theta_transform), so +yaw is
 * a rotation about world +Y by -yaw, and +pitch tilts the view up. */
void camera_apply_look(float R[3][3], float yaw_deg, float pitch_deg) {
    const float d2r = 3.14159265358979f / 180.0f;
    if (yaw_deg != 0.0f) {
        const float b = -yaw_deg * d2r;
        const float cb = cosf(b), sb = sinf(b);
        const float E[3][3] = { { cb, 0.0f, sb }, { 0.0f, 1.0f, 0.0f }, { -sb, 0.0f, cb } };
        m3_mul(R, E, R);
    }
    if (pitch_deg != 0.0f) {
        const float fx = -R[0][2], fz = -R[2][2];
        float ax = -fz, az = fx;    /* forward x world up */
        const float len = sqrtf(ax * ax + az * az);
        if (len > 1e-4f) {
            ax /= len;
            az /= len;
            const float g = pitch_deg * d2r;
            const float c = cosf(g), s = sinf(g), k = 1.0f - c;
            /* Rodrigues, axis (ax, 0, az) */
            const float E[3][3] = {
                { c + ax * ax * k, -az * s, ax * az * k },
                { az * s, c, -ax * s },
                { az * ax * k, ax * s, c + az * az * k },
            };
            m3_mul(R, E, R);
        }
    }
}

/* sky.c clips each sky fan to this tick's screen edges (skyClamp). Turned
 * toward another view, a vertex on an edge moves inward and uncovers the clear
 * colour. Put such vertices back on their edge with the texture coordinates
 * and w the sky plane has there: for a planar polygon s/w, t/w and 1/w are
 * affine in screen space, so they are fitted from three of the turned
 * vertices and evaluated at the edge point. */
void sky_refill_edges(const PortFloatVtx *orig, PortFloatVtx *o, size_t n) {
    /* skyClamp stops a quarter pixel short of the right and bottom edges
     * ((left + width) * 4 - 1): 0.9984 in NDC on a 320-wide view */
    const float e = 0.995f;
    auto on_edge = [&](size_t i, int axis, int side) {
        const float c = axis ? orig[i].y : orig[i].x;
        return side < 0 ? c <= -e * orig[i].w : c >= e * orig[i].w;
    };
    bool need = false;
    for (size_t i = 0; i < n; i++) {
        if (orig[i].w > 0.0f && o[i].w > 0.0f &&
            ((on_edge(i, 0, -1) && o[i].x > -o[i].w) || (on_edge(i, 0, 1) && o[i].x < o[i].w) ||
             (on_edge(i, 1, -1) && o[i].y > -o[i].w) || (on_edge(i, 1, 1) && o[i].y < o[i].w))) {
            need = true;
        }
    }
    if (!need) {
        return;
    }
    size_t t[3];
    float det = 0.0f;
    bool found = false;
    for (size_t a = 0; a < n && !found; a++) {
        for (size_t b = a + 1; b < n && !found; b++) {
            for (size_t c = b + 1; c < n && !found; c++) {
                if (o[a].w <= 0.0f || o[b].w <= 0.0f || o[c].w <= 0.0f) {
                    continue;
                }
                const float xa = o[a].x / o[a].w, ya = o[a].y / o[a].w;
                const float xb = o[b].x / o[b].w, yb = o[b].y / o[b].w;
                const float xc = o[c].x / o[c].w, yc = o[c].y / o[c].w;
                det = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya);
                if (fabsf(det) > 1e-4f) {
                    t[0] = a; t[1] = b; t[2] = c;
                    found = true;
                }
            }
        }
    }
    if (!found) {
        return;
    }
    float X[3], Y[3], A[3][3];   /* A[attr][vertex]: s/w, t/w, 1/w */
    for (int j = 0; j < 3; j++) {
        const PortFloatVtx &p = o[t[j]];
        X[j] = p.x / p.w;
        Y[j] = p.y / p.w;
        A[0][j] = p.s / p.w;
        A[1][j] = p.t / p.w;
        A[2][j] = 1.0f / p.w;
    }
    float C[3][3];   /* attr = C0 + C1 X + C2 Y */
    for (int k = 0; k < 3; k++) {
        const float d1 = A[k][1] - A[k][0], d2 = A[k][2] - A[k][0];
        C[k][1] = (d1 * (Y[2] - Y[0]) - d2 * (Y[1] - Y[0])) / det;
        C[k][2] = ((X[1] - X[0]) * d2 - (X[2] - X[0]) * d1) / det;
        C[k][0] = A[k][0] - C[k][1] * X[0] - C[k][2] * Y[0];
    }
    for (size_t i = 0; i < n; i++) {
        if (orig[i].w <= 0.0f || o[i].w <= 0.0f) {
            continue;
        }
        float nx = o[i].x / o[i].w, ny = o[i].y / o[i].w;
        bool moved = false;
        if (on_edge(i, 0, -1) && nx > -1.0f) { nx = -1.0f; moved = true; }
        if (on_edge(i, 0, 1) && nx < 1.0f) { nx = 1.0f; moved = true; }
        if (on_edge(i, 1, -1) && ny > -1.0f) { ny = -1.0f; moved = true; }
        if (on_edge(i, 1, 1) && ny < 1.0f) { ny = 1.0f; moved = true; }
        if (!moved) {
            continue;
        }
        /* The fan's lower corners are its far cut, just above the plane's
         * horizon, and skyClamp leaves the corners only roughly coplanar: the
         * fitted plane can put the edge point at or past its horizon. Keep the
         * point (the fan below shares it) and the vertex's own depth then. */
        const float q0 = 1.0f / o[i].w;
        float q = C[2][0] + C[2][1] * nx + C[2][2] * ny;
        if (!(q > 0.25f * q0)) {
            q = q0;
        }
        const float w = 1.0f / q;
        o[i].s = (C[0][0] + C[0][1] * nx + C[0][2] * ny) * w;
        o[i].t = (C[1][0] + C[1][1] * nx + C[1][2] * ny) * w;
        o[i].x = nx * w;
        o[i].y = ny * w;
        o[i].w = w;
    }
}

/* -------------------------------------------------------------- frames --- */

enum {
    RF_PV = 1,      /* drawn under the level's projection x view (bg rooms) */
    RF_ANCHOR = 2,  /* z-buffer off: attached to the camera (first-person gun) */
    RF_DYN = 4,     /* vertex source rebuilt every tick */
    RF_VIEWMODEL = 8, /* drawn under the gun's perspective-normalize value */
};

struct Rec {
    uintptr_t src;
    uint32_t n;
    int8_t cam;     /* scene camera, -1 = other (menus, watch, HUD) */
    uint8_t flags;
    bool aff_ok;
    int32_t inst;
    int32_t prev;   /* matched record in the previous display list, -1 */
    M4 C;           /* camera-space transform (others: the modelview) */
    M4 W;           /* world transform, origin at this tick's camera (scene cameras only) */
    M4 Pslot;       /* the loaded projection (others only) */
    Aff aff;        /* decomposition of what gets blended: W, or C if anchored/other */
    float pos[3];   /* scene: the batch origin in absolute world coordinates */
};

struct Inst {
    uintptr_t base;
    uintptr_t sig;
    int cam;
    float anchor[3];
    int prev;
    std::vector<int> recs;
};

struct MtxRec {
    M4 m;
    int8_t cam;
    uint8_t pv;
    uint8_t inst;
};

struct CamInfo {
    bool ok;
    Aff a;          /* rotation + scale of V */
    float pos[3];   /* camera position */
    float rot[3][3];/* normalized rotation of V */
};

struct RecKey {
    uintptr_t src;
    int32_t cam;
    int32_t occ;
    bool operator==(const RecKey &o) const { return src == o.src && cam == o.cam && occ == o.occ; }
};
struct RecKeyHash {
    size_t operator()(const RecKey &k) const {
        uint64_t h = (uint64_t)k.src * 0x9E3779B97F4A7C15ull;
        h ^= ((uint64_t)(uint32_t)k.cam << 40) ^ (uint64_t)(uint32_t)k.occ;
        return (size_t)(h ^ (h >> 29));
    }
};

struct Frame {
    bool valid;
    GfxInterpFrame info;
    CamInfo cam[GFX_INTERP_MAX_CAMS];
    bool camok[GFX_INTERP_MAX_CAMS];
    M4 Vinv[GFX_INTERP_MAX_CAMS];
    M4 V0inv[GFX_INTERP_MAX_CAMS];   /* inverse of V's rotation and scale (no translation) */
    M4 Pinv[GFX_INTERP_MAX_CAMS];
    std::vector<Rec> recs;
    std::vector<MtxRec> mtx;
    std::vector<Inst> insts;
    std::unordered_map<uintptr_t, int> inst_by_base;
    std::unordered_map<RecKey, std::vector<int>, RecKeyHash> inst_by_sig;  /* occ unused (0) */
    std::unordered_map<RecKey, int, RecKeyHash> rec_by_key;
    std::unordered_map<RecKey, int, RecKeyHash> occ;                     /* occ unused (0) */

    void reset() {
        valid = false;
        recs.clear();
        mtx.clear();
        insts.clear();
        inst_by_base.clear();
        inst_by_sig.clear();
        rec_by_key.clear();
        occ.clear();
    }
};

enum Mode { MODE_OFF, MODE_RECORD, MODE_REPLAY };

Frame g_fr[2];
int g_cur = 0;
Mode g_mode = MODE_OFF;
GfxInterpPass g_pass;
bool g_in_pass = false;
bool g_replay_ok = false;
bool g_diverged = false;
double g_last_dl_ms = 0.0;
std::chrono::steady_clock::time_point g_pass_t0;

/* frame-level (set by gfx_interp_new_frame) */
bool g_frame_blend = false;
int g_dticks = 1;
bool g_cam_cut[GFX_INTERP_MAX_CAMS];
std::vector<char> g_prev_claimed;

/* pass cursor */
size_t g_mtx_ord, g_rec_ord;
int g_proj_cam = -1;
bool g_proj_pv = false;
M4 g_vbg;               /* projection slot * P^-1 for the current PV projection */
bool g_mv_inst = false;
int g_cur_inst = -1;
uintptr_t g_seg3 = 0;
uint32_t g_persp_cur = 0;                        /* current G_MW_PERSPNORM value */
uint32_t g_persp_scene[GFX_INTERP_MAX_CAMS];     /* each camera's scene value */
bool g_persp_scene_set[GFX_INTERP_MAX_CAMS];

/* pass, per camera */
bool g_cam_changed[GFX_INTERP_MAX_CAMS];
bool g_proj_changed[GFX_INTERP_MAX_CAMS];
M4 g_Vd[GFX_INTERP_MAX_CAMS], g_Pd[GFX_INTERP_MAX_CAMS];
/* the displayed view for camera-relative W: origin at this tick's camera */
M4 g_Vdr[GFX_INTERP_MAX_CAMS], g_VPdr[GFX_INTERP_MAX_CAMS];
float g_sky[GFX_INTERP_MAX_CAMS][3][3];
float g_cam_rel[GFX_INTERP_MAX_CAMS][3];   /* displayed camera position - this tick's (world) */
PortFloatVtx g_sky_buf[256];

/* GE_INTERP_LOG=1: one line per second of matching / timing stats. */
int g_log = -1;
struct {
    uint64_t t0;
    unsigned frames, passes, recs, matched, inst, inst_matched, cuts, diverged, cam_recs, cam_unmatched, anchored;
    unsigned bg_matched, bg_err_n;
    double dl_ms, bg_drift, bg_err;
    float bg_drift_max, bg_err_max;
    float yaw_seen, yaw_applied, pitch_seen, pitch_applied;
} g_st;

Frame &cur() { return g_fr[g_cur]; }
Frame &prv() { return g_fr[g_cur ^ 1]; }

bool is_dyn(const Frame &f, uintptr_t p) {
    for (int i = 0; i < 2; i++) {
        if (p >= f.info.dyn_lo[i] && p < f.info.dyn_hi[i]) {
            return true;
        }
    }
    return false;
}

bool cam_info_from(const GfxInterpCam &gc, CamInfo &ci, M4 Vinv, M4 V0inv) {
    ci.ok = false;
    if (!gc.valid || !m4_invert(Vinv, gc.V) || !aff_decompose(gc.V, ci.a)) {
        return false;
    }
    M4 V0;
    m4_copy(V0, gc.V);
    V0[3][0] = V0[3][1] = V0[3][2] = 0.0f;
    if (!m4_invert(V0inv, V0)) {
        return false;
    }
    for (int i = 0; i < 3; i++) {
        ci.pos[i] = Vinv[3][i];
    }
    for (int i = 0; i < 3; i++) {
        const float s = ci.a.s[i];
        for (int j = 0; j < 3; j++) {
            ci.rot[i][j] = gc.V[i][j] / s;
        }
    }
    ci.ok = true;
    return true;
}

void stats_tick(void) {
    if (g_log < 0) {
        const char *v = getenv("GE_INTERP_LOG");
        g_log = v ? (atoi(v) > 1 ? 2 : 1) : 0;   /* 2: also one line per draw (libultra.c) */
    }
    if (!g_log) {
        return;
    }
    const uint64_t now = sysGetMicroseconds();
    if (g_st.t0 == 0) {
        g_st.t0 = now;
        return;
    }
    if (now - g_st.t0 < 1000000) {
        return;
    }
    const unsigned f = g_st.frames ? g_st.frames : 1;
    sysLogPrintf(LOG_NOTE,
                 "D573 interp: frames=%u passes=%u (%.2f/frame) recs/frame=%u matched=%.1f%% "
                 "scene-unmatched=%.1f%% anchored/frame=%u inst=%u matched=%.1f%% cuts=%u diverged=%u dl=%.2fms "
                 "yaw seen/applied=%.2f/%.2f pitch seen/applied=%.2f/%.2f bg-drift mrad mean/max=%.3f/%.3f "
                 "bg-err px mean/max=%.3f/%.3f",
                 g_st.frames, g_st.passes, (double)g_st.passes / f, g_st.recs / f,
                 g_st.recs ? 100.0 * g_st.matched / g_st.recs : 0.0,
                 g_st.cam_recs ? 100.0 * g_st.cam_unmatched / g_st.cam_recs : 0.0,
                 g_st.anchored / f, g_st.inst / f, g_st.inst ? 100.0 * g_st.inst_matched / g_st.inst : 0.0,
                 g_st.cuts, g_st.diverged, g_st.dl_ms / f, (double)g_st.yaw_seen, (double)g_st.yaw_applied,
                 (double)g_st.pitch_seen, (double)g_st.pitch_applied,
                 g_st.bg_matched ? g_st.bg_drift / g_st.bg_matched : 0.0, (double)g_st.bg_drift_max,
                 g_st.bg_err_n ? g_st.bg_err / g_st.bg_err_n : 0.0, (double)g_st.bg_err_max);
    memset(&g_st, 0, sizeof(g_st));
    g_st.t0 = now;
}

/* GE_INTERP_LOG=2 trace of camera 0's terms for the current draw */
bool g_dbg_blend;
float g_dbg_h0, g_dbg_extra;
float g_dbg_gunx = 0.0f;   /* screen x of the first view-model batch of a draw */
bool g_dbg_gun_seen = false;

/* Per-pass camera setup: the blended (and mouse-turned) view of every camera. */
void setup_pass_cams(void) {
    Frame &c = cur();
    Frame &p = prv();
    const float a = g_pass.alpha < 0.0f ? 0.0f : (g_pass.alpha > 1.0f ? 1.0f : g_pass.alpha);
    for (int k = 0; k < GFX_INTERP_MAX_CAMS; k++) {
        g_cam_changed[k] = false;
        g_proj_changed[k] = false;
        if (!c.camok[k]) {
            continue;
        }
        const GfxInterpCam &cc = c.info.cams[k];
        const bool blend = g_frame_blend && p.valid && p.camok[k] && !g_cam_cut[k] && a < 1.0f &&
                           (memcmp(p.info.cams[k].V, cc.V, sizeof(M4)) != 0 ||
                            memcmp(p.info.cams[k].P, cc.P, sizeof(M4)) != 0);
        float q[4], s[3], pos[3];
        if (blend) {
            q_slerp(q, p.cam[k].a.q, c.cam[k].a.q, a);
            for (int i = 0; i < 3; i++) {
                s[i] = p.cam[k].a.s[i] + (c.cam[k].a.s[i] - p.cam[k].a.s[i]) * a;
                pos[i] = p.cam[k].pos[i] + (c.cam[k].pos[i] - p.cam[k].pos[i]) * a;
            }
        } else {
            memcpy(q, c.cam[k].a.q, sizeof(q));
            memcpy(s, c.cam[k].a.s, sizeof(s));
            memcpy(pos, c.cam[k].pos, sizeof(pos));
        }
        float R[3][3];
        m3_from_quat(q, R);
        bool turned = false;
        float dbg_yaw = 0.0f;
        const float dbg_h0 = atan2f(R[0][2], -R[2][2]) * (180.0f / 3.14159265358979f);
        if (g_pass.predict && k == c.info.look_cam) {
            /* The mouse part of this tick's rotation is shown at once, not
             * blended in; mouse motion the game has not read yet is added. */
            const float yaw = g_pass.pend_yaw_deg + (blend ? (1.0f - a) * c.info.look_yaw_deg : 0.0f);
            const float pitch = g_pass.pend_pitch_deg + (blend ? (1.0f - a) * c.info.look_pitch_deg : 0.0f);
            dbg_yaw = yaw;
            if (fabsf(yaw) > 1e-4f || fabsf(pitch) > 1e-4f) {
                camera_apply_look(R, yaw, pitch);
                turned = true;
            }
        }
        if (k == 0) {
            g_dbg_blend = blend;
            g_dbg_h0 = dbg_h0;
            g_dbg_extra = dbg_yaw;
        }
        if (!blend && !turned) {
            m4_copy(g_Vd[k], cc.V);
            m4_copy(g_Pd[k], cc.P);
            m4_copy(g_Vdr[k], cc.V);
            g_Vdr[k][3][0] = g_Vdr[k][3][1] = g_Vdr[k][3][2] = 0.0f;
            m4_mul(g_VPdr[k], g_Vdr[k], g_Pd[k]);
            continue;
        }
        M4 &V = g_Vd[k];
        M4 &Vr = g_Vdr[k];
        float *rel = g_cam_rel[k];
        for (int i = 0; i < 3; i++) {
            rel[i] = pos[i] - c.cam[k].pos[i];
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                V[i][j] = Vr[i][j] = R[i][j] * s[i];
            }
            V[i][3] = Vr[i][3] = 0.0f;
        }
        for (int j = 0; j < 3; j++) {
            V[3][j] = -(pos[0] * V[0][j] + pos[1] * V[1][j] + pos[2] * V[2][j]);
            Vr[3][j] = -(rel[0] * V[0][j] + rel[1] * V[1][j] + rel[2] * V[2][j]);
        }
        V[3][3] = Vr[3][3] = 1.0f;
        if (blend) {
            m4_lerp(g_Pd[k], p.info.cams[k].P, cc.P, a);
            g_proj_changed[k] = memcmp(p.info.cams[k].P, cc.P, sizeof(M4)) != 0;
        } else {
            m4_copy(g_Pd[k], cc.P);
        }
        m4_mul(g_VPdr[k], Vr, g_Pd[k]);
        g_cam_changed[k] = true;
        /* sky: camera-space directions of this tick -> the displayed view */
        float Rt[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Rt[i][j] = c.cam[k].rot[j][i];
            }
        }
        m3_mul(g_sky[k], Rt, R);
    }
    if (g_log >= 2 && c.camok[0]) {
        /* GE_INTERP_LOG=2: the view this draw shows (heading degrees, position) */
        M4 inv;
        if (m4_invert(inv, g_Vd[0])) {
            const float h = atan2f(g_Vd[0][0][2], -g_Vd[0][2][2]) * (180.0f / 3.14159265358979f);
            const float sc = c.cam[0].a.s[1] > 0.0f ? c.cam[0].a.s[1] : 1.0f;   /* stage visibility scale */
            const float pt = asinf(fmaxf(-1.0f, fminf(1.0f, -g_Vd[0][1][2] / sc))) * (180.0f / 3.14159265358979f);
            sysLogPrintf(LOG_NOTE, "D573 view h=%.4f p=%.4f pos=%.3f,%.3f,%.3f a=%.3f blended_h=%.4f extra=%.4f blend=%d m=%.4f pend=%.4f prevgunx=%.5f",
                         (double)h, (double)pt, (double)inv[3][0], (double)inv[3][1], (double)inv[3][2], (double)a,
                         (double)g_dbg_h0, (double)g_dbg_extra, (int)g_dbg_blend,
                         (double)c.info.look_yaw_deg, (double)g_pass.pend_yaw_deg, (double)g_dbg_gunx);
            g_dbg_gun_seen = false;
        }
    }
}

/* A scene batch's camera-space transform C -> W, its world transform with the
 * origin at the camera (campos), and pos, its origin in world coordinates.
 * Level geometry's C carries a homogeneous scale (the fixed-point projection x
 * view over the fixed-point projection: ~1e-3 off, different every tick);
 * dividing it out changes nothing on screen. */
void scene_world(M4 C, M4 W, float pos[3], const M4 V0inv, const float campos[3]) {
    const float w = C[3][3];
    if (w > 1e-6f && w != 1.0f) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                C[i][j] /= w;
            }
        }
    }
    m4_mul(W, C, V0inv);
    for (int j = 0; j < 3; j++) {
        pos[j] = W[3][j] + campos[j];
    }
}

/* poff (world transforms only): the previous tick's camera position minus
 * this tick's, which moves the previous W to this tick's origin. */
void blend_pair(M4 out, const Rec &p, const Rec &c, bool use_w, float a, const float *poff) {
    if (p.aff_ok && c.aff_ok && (p.aff.s[0] > 0.0f) == (c.aff.s[0] > 0.0f)) {
        Aff pa = p.aff, b;
        if (poff) {
            for (int j = 0; j < 3; j++) {
                pa.t[j] += poff[j];
            }
        }
        aff_blend(b, pa, c.aff, a);
        aff_compose(out, b);
    } else if (use_w) {
        M4 pw;
        m4_copy(pw, p.W);
        if (poff) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    pw[i][j] += pw[i][3] * poff[j];
                }
            }
        }
        m4_lerp(out, pw, c.W, a);
    } else {
        m4_lerp(out, p.C, c.C, a);
    }
}

bool rec_blend_impl(const Rec &r, float mp[4][4], float mv[4][4], bool *mvchg);

bool rec_blend(const Rec &r, float mp[4][4], float mv[4][4], bool *mvchg) {
    const bool changed = rec_blend_impl(r, mp, mv, mvchg);
    if ((r.flags & RF_VIEWMODEL) && !g_dbg_gun_seen && g_log >= 2) {   /* trace: gun screen x */
        g_dbg_gun_seen = true;
        float m[4][4];
        if (changed) {
            m4_copy(m, mp);
        } else if (r.cam >= 0) {
            m4_mul(m, r.C, cur().info.cams[r.cam].P);
        } else {
            m4_mul(m, r.C, r.Pslot);
        }
        g_dbg_gunx = m[3][3] != 0.0f ? m[3][0] / m[3][3] : 0.0f;
    }
    return changed;
}

bool rec_blend_impl(const Rec &r, float mp[4][4], float mv[4][4], bool *mvchg) {
    const float a = g_pass.alpha < 0.0f ? 0.0f : g_pass.alpha;
    const Rec *p = (r.prev >= 0 && g_frame_blend && a < 1.0f) ? &prv().recs[r.prev] : nullptr;
    if (r.cam >= 0) {
        const int k = r.cam;
        if (p && g_cam_cut[k]) {
            p = nullptr;
        }
        if (r.flags & RF_ANCHOR) {
            if (!p && !g_proj_changed[k]) {
                return false;
            }
            M4 C;
            if (p) {
                blend_pair(C, *p, r, false, a, nullptr);
            } else {
                m4_copy(C, r.C);
            }
            m4_mul(mp, C, g_Pd[k]);
            if (!(r.flags & RF_PV)) {
                m4_copy(mv, C);
                *mvchg = true;
            }
            return true;
        }
        if (!p && !g_cam_changed[k]) {
            return false;
        }
        M4 W;
        if (p) {
            const float *pc = prv().cam[k].pos, *cc = cur().cam[k].pos;
            const float poff[3] = { pc[0] - cc[0], pc[1] - cc[1], pc[2] - cc[2] };
            blend_pair(W, *p, r, true, a, poff);
        } else {
            m4_copy(W, r.W);
        }
        m4_mul(mp, W, g_VPdr[k]);
        if (p && g_log > 0 && (r.flags & RF_PV)) {
            /* static level geometry: where the blend put the batch origin vs
             * where it is seen from the displayed camera (1080p pixels) */
            const float *b = r.W[3];
            float ci[4];
            for (int j = 0; j < 4; j++) {
                ci[j] = b[0] * g_VPdr[k][0][j] + b[1] * g_VPdr[k][1][j] + b[2] * g_VPdr[k][2][j] + b[3] * g_VPdr[k][3][j];
            }
            const float ix = ci[0] / ci[3], iy = ci[1] / ci[3];
            if (mp[3][3] > 1.0f && ci[3] > 1.0f && fabsf(ix) < 1.5f && fabsf(iy) < 1.5f) {   /* origin on screen */
                const float dx = mp[3][0] / mp[3][3] - ix, dy = mp[3][1] / mp[3][3] - iy;
                const float e = 540.0f * sqrtf(dx * dx + dy * dy);
                g_st.bg_err += e;
                g_st.bg_err_max = fmaxf(g_st.bg_err_max, e);
                g_st.bg_err_n++;
            }
        }
        if (!(r.flags & RF_PV)) {
            m4_mul(mv, W, g_Vdr[k]);
            *mvchg = true;
        }
        return true;
    }
    if (!p) {
        return false;
    }
    M4 C, P;
    blend_pair(C, *p, r, false, a, nullptr);
    m4_lerp(P, p->Pslot, r.Pslot, a);
    m4_mul(mp, C, P);
    m4_copy(mv, C);
    *mvchg = true;
    return true;
}

bool rec_compatible(const Rec &p, const Rec &c) {
    if (p.n != c.n || p.cam != c.cam) {
        return false;
    }
    if ((p.flags & (RF_PV | RF_ANCHOR)) != (c.flags & (RF_PV | RF_ANCHOR))) {
        return false;
    }
    if (p.src != c.src && !((p.flags & RF_DYN) && (c.flags & RF_DYN))) {
        return false;
    }
    const bool world = c.cam >= 0 && !(c.flags & RF_ANCHOR);
    if (world && dist3(p.pos, c.pos) > kObjSnapDist * (float)g_dticks) {
        return false;
    }
    if (p.aff_ok && c.aff_ok) {
        if ((p.aff.s[0] > 0.0f) != (c.aff.s[0] > 0.0f)) {
            return false;
        }
        if (!world && dist3(p.aff.t, c.aff.t) > kObjSnapDist * (float)g_dticks) {
            return false;
        }
        if (q_angle(p.aff.q, c.aff.q) > kObjSnapAngle) {
            return false;
        }
    }
    return true;
}

int match_instance(const Inst &ci) {
    Frame &p = prv();
    auto it = p.inst_by_sig.find(RecKey{ ci.sig, ci.cam, 0 });
    if (it == p.inst_by_sig.end()) {
        return -1;
    }
    int best = -1;
    float bestd = kInstMatchDist * (float)g_dticks;
    for (int j : it->second) {
        if (g_prev_claimed[j]) {
            continue;
        }
        const float d = dist3(p.insts[j].anchor, ci.anchor);
        if (d < bestd) {
            bestd = d;
            best = j;
        }
    }
    if (best >= 0) {
        g_prev_claimed[best] = 1;
    }
    return best;
}

/* First draw of a display list: build this batch's record and its match. */
Rec &record_vertex(uintptr_t src, size_t n, const float mv[4][4], const float pm[4][4],
                   uint32_t geometry_mode, uintptr_t seg5) {
    Frame &c = cur();
    Frame &p = prv();
    c.recs.emplace_back();
    const int idx = (int)c.recs.size() - 1;
    Rec &r = c.recs.back();
    r.src = src;
    r.n = (uint32_t)n;
    r.flags = 0;
    r.inst = -1;
    r.prev = -1;
    if (is_dyn(c, src)) {
        r.flags |= RF_DYN;
    }
    r.cam = (int8_t)((g_proj_cam >= 0 && c.camok[g_proj_cam]) ? g_proj_cam : -1);
    /* Camera-attached: the first-person gun and hands, drawn under their own
     * perspective-normalize value (and anything with the z-buffer off). They
     * must not take the camera's blend or the predicted mouse turn. */
    const bool viewmodel = r.cam >= 0 && g_persp_scene_set[r.cam] && g_persp_cur != g_persp_scene[r.cam];
    if (viewmodel) {
        r.flags |= RF_VIEWMODEL;
    }
    if ((viewmodel || !(geometry_mode & G_ZBUFFER)) && !(r.cam >= 0 && g_proj_pv)) {
        r.flags |= RF_ANCHOR;
    }
    if (r.cam >= 0) {
        if (g_proj_pv) {
            r.flags |= RF_PV;
            m4_mul(r.C, mv, g_vbg);
        } else {
            m4_copy(r.C, mv);
        }
        scene_world(r.C, r.W, r.pos, c.V0inv[r.cam], c.cam[r.cam].pos);
        r.aff_ok = aff_decompose((r.flags & RF_ANCHOR) ? r.C : r.W, r.aff);
    } else {
        m4_copy(r.C, mv);
        m4_copy(r.Pslot, pm);
        r.aff_ok = aff_decompose(r.C, r.aff);
    }
    const bool can_match = g_frame_blend && p.valid;
    const float *anchor = (r.cam >= 0 && !(r.flags & RF_ANCHOR)) ? r.pos : r.C[3];

    if (g_mv_inst && g_cur_inst >= 0) {
        r.inst = g_cur_inst;
        Inst &I = c.insts[g_cur_inst];
        if (I.recs.empty()) {
            I.sig = seg5;
            I.cam = r.cam;
            memcpy(I.anchor, anchor, sizeof(I.anchor));
            I.prev = can_match ? match_instance(I) : -1;
            c.inst_by_sig[RecKey{ I.sig, I.cam, 0 }].push_back(g_cur_inst);
            g_st.inst++;
            if (I.prev >= 0) {
                g_st.inst_matched++;
            }
        }
        const size_t ord = I.recs.size();
        I.recs.push_back(idx);
        if (I.prev >= 0) {
            const Inst &PI = p.insts[I.prev];
            if (ord < PI.recs.size() && rec_compatible(p.recs[PI.recs[ord]], r)) {
                r.prev = PI.recs[ord];
            }
        }
    } else if (!(r.flags & RF_DYN)) {
        int &o = c.occ[RecKey{ src, r.cam, 0 }];
        const RecKey key{ src, r.cam, o++ };
        c.rec_by_key[key] = idx;
        if (can_match) {
            auto it = p.rec_by_key.find(key);
            if (it != p.rec_by_key.end() && rec_compatible(p.recs[it->second], r)) {
                r.prev = it->second;
            }
        }
    }
    g_st.recs++;
    if (r.prev >= 0) {
        g_st.matched++;
        if (g_log > 0 && (r.flags & (RF_PV | RF_ANCHOR)) == RF_PV && !g_cam_cut[r.cam]) {
            /* Level geometry is static: the direction from this tick's camera
             * to a batch's origin should be the same whichever tick's
             * transform placed it (milliradians, ~pixels at 1080p). */
            const float *cp = c.cam[r.cam].pos, *a0 = p.recs[r.prev].pos;
            const float u[3] = { a0[0] - cp[0], a0[1] - cp[1], a0[2] - cp[2] };
            const float v[3] = { r.pos[0] - cp[0], r.pos[1] - cp[1], r.pos[2] - cp[2] };
            const float x[3] = { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };
            const float zero[3] = { 0.0f, 0.0f, 0.0f };
            const float uv = dist3(u, zero) * dist3(v, zero);
            const float d = uv > 1e-3f ? 1000.0f * dist3(x, zero) / uv : 0.0f;
            g_st.bg_drift += d;
            g_st.bg_drift_max = fmaxf(g_st.bg_drift_max, d);
            g_st.bg_matched++;
        }
    }
    if (r.cam >= 0 && !(r.flags & RF_ANCHOR)) {
        g_st.cam_recs++;
        if (r.prev < 0) {
            g_st.cam_unmatched++;
        }
    }
    if (r.cam >= 0 && (r.flags & RF_ANCHOR)) {
        g_st.anchored++;
    }
    return r;
}

}  // namespace

/* ------------------------------------------------------------ C API --- */

extern "C" void gfx_interp_new_frame(const GfxInterpFrame *f, const GfxInterpPass *pass) {
    stats_tick();
    g_cur ^= 1;
    Frame &c = cur();
    Frame &p = prv();
    c.reset();
    c.info = *f;
    c.valid = true;
    for (int k = 0; k < GFX_INTERP_MAX_CAMS; k++) {
        c.camok[k] = cam_info_from(f->cams[k], c.cam[k], c.Vinv[k], c.V0inv[k]) && m4_invert(c.Pinv[k], f->cams[k].P);
    }
    g_dticks = f->sim_tick - p.info.sim_tick;
    g_frame_blend = p.valid && p.info.stage == f->stage && g_dticks >= 1 && g_dticks <= kMaxTicksBlend;
    for (int k = 0; k < GFX_INTERP_MAX_CAMS; k++) {
        g_cam_cut[k] = false;
        if (!g_frame_blend || !c.camok[k] || !p.camok[k]) {
            continue;
        }
        if (dist3(p.cam[k].pos, c.cam[k].pos) > kCamCutDist * (float)g_dticks ||
            q_angle(p.cam[k].a.q, c.cam[k].a.q) > kCamCutAngle) {
            g_cam_cut[k] = true;
            g_st.cuts++;
        }
    }
    if (g_log > 0 && g_frame_blend && f->look_cam >= 0 && c.camok[f->look_cam] && p.camok[f->look_cam]) {
        /* yaw check: the heading change the game produced vs the mouse yaw it applied */
        const int k = f->look_cam;
        const float hp = atan2f(p.cam[k].rot[0][2], -p.cam[k].rot[2][2]);
        const float hc = atan2f(c.cam[k].rot[0][2], -c.cam[k].rot[2][2]);
        float dh = (hc - hp) * (180.0f / 3.14159265358979f);
        if (dh > 180.0f) dh -= 360.0f;
        if (dh < -180.0f) dh += 360.0f;
        g_st.yaw_seen += dh;
        g_st.yaw_applied += f->look_yaw_deg;
        /* pitch: the view's forward is -column 2; up = +Y */
        const float pp = asinf(fmaxf(-1.0f, fminf(1.0f, -p.cam[k].rot[1][2])));
        const float pc = asinf(fmaxf(-1.0f, fminf(1.0f, -c.cam[k].rot[1][2])));
        g_st.pitch_seen += (pc - pp) * (180.0f / 3.14159265358979f);
        g_st.pitch_applied += f->look_pitch_deg;
    }
    g_prev_claimed.assign(p.insts.size(), 0);
    g_pass = *pass;
    g_mode = MODE_RECORD;
    g_replay_ok = false;
    g_st.frames++;
}

extern "C" void gfx_interp_replay(const GfxInterpPass *pass) {
    if (cur().valid && g_replay_ok) {
        g_pass = *pass;
        g_mode = MODE_REPLAY;
    } else {
        g_mode = MODE_OFF;
    }
}

extern "C" void gfx_interp_disable(void) {
    g_mode = MODE_OFF;
    g_replay_ok = false;
    g_fr[0].reset();
    g_fr[1].reset();
}

extern "C" int gfx_interp_can_replay(void) {
    return g_replay_ok && cur().valid;
}

extern "C" double gfx_interp_last_dl_ms(void) {
    return g_last_dl_ms;
}

extern "C" int gfx_interp_log_level(void) {
    return g_log > 0 ? g_log : 0;
}

/* -------------------------------------------------------------- hooks --- */

bool interp_hooks_active(void) {
    return g_in_pass && !g_diverged;
}

int interp_pass_begin(void) {
    g_in_pass = false;
    if (g_mode == MODE_OFF) {
        return -1;
    }
    g_in_pass = true;
    g_diverged = false;
    g_mtx_ord = 0;
    g_rec_ord = 0;
    g_proj_cam = -1;
    g_proj_pv = false;
    g_mv_inst = false;
    g_cur_inst = -1;
    g_seg3 = 0;
    g_persp_cur = 0;
    for (int k = 0; k < GFX_INTERP_MAX_CAMS; k++) {
        g_persp_scene_set[k] = false;
    }
    setup_pass_cams();
    g_pass_t0 = std::chrono::steady_clock::now();
    g_st.passes++;
    return g_mode == MODE_REPLAY ? 1 : 0;
}

void interp_pass_end(bool drew) {
    if (!g_in_pass) {
        return;
    }
    g_in_pass = false;
    if (g_mode == MODE_RECORD) {
        g_last_dl_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_pass_t0).count();
        g_st.dl_ms += g_last_dl_ms;
        if (!drew) {
            /* nothing walked: no history to match against or replay */
            cur().valid = false;
        }
        g_replay_ok = drew && !g_diverged && cur().valid;
    } else if (g_mode == MODE_REPLAY) {
        if (g_diverged || g_rec_ord != cur().recs.size()) {
            g_replay_ok = false;
            g_st.diverged++;
        }
    }
    g_diverged = false;
}

void interp_on_matrix(uint8_t params, const void *addr, uintptr_t w1, float matrix[4][4]) {
    Frame &c = cur();
    if (g_mode == MODE_REPLAY) {
        if (g_mtx_ord >= c.mtx.size()) {
            g_diverged = true;
            return;
        }
        const MtxRec &m = c.mtx[g_mtx_ord++];
        m4_copy(matrix, m.m);
        if (params & G_MTX_PROJECTION) {
            g_proj_cam = m.cam;
            g_proj_pv = m.pv != 0;
        }
        return;
    }
    c.mtx.emplace_back();
    MtxRec &m = c.mtx.back();
    m4_copy(m.m, matrix);
    m.cam = -1;
    m.pv = 0;
    m.inst = 0;
    const uintptr_t a = norm((uintptr_t)addr);
    if (params & G_MTX_PROJECTION) {
        if (params & G_MTX_LOAD) {
            for (int k = 0; k < GFX_INTERP_MAX_CAMS; k++) {
                if (!c.camok[k]) {
                    continue;
                }
                if (a == norm((uintptr_t)c.info.cams[k].proj)) {
                    m.cam = (int8_t)k;
                    break;
                }
                if (a == norm((uintptr_t)c.info.cams[k].pv)) {
                    m.cam = (int8_t)k;
                    m.pv = 1;
                    break;
                }
            }
        }
        g_proj_cam = m.cam;
        g_proj_pv = m.pv != 0;
        if (m.pv) {
            m4_mul(g_vbg, matrix, c.Pinv[m.cam]);
        }
    } else {
        /* a model's matrices: segment 3 (render_pos), referenced through the
         * segment by the model's own display lists or directly (the two-matrix
         * skin seams, model.c) */
        const unsigned seg = (unsigned)((w1 >> 24) & 0xf);
        const bool segref = seg == 3 && (w1 < 0x10000000u || (w1 & 1));
        const bool direct = g_seg3 != 0 && a >= g_seg3 && a < g_seg3 + kInstMtxSpan && is_dyn(c, a);
        g_mv_inst = g_cur_inst >= 0 && (segref || direct);
        m.inst = g_mv_inst;
    }
}

void interp_on_perspnorm(uint32_t value) {
    if (g_mode != MODE_RECORD) {
        return;   /* replays use the recorded classification */
    }
    g_persp_cur = value;
    /* the scene's value is set right after its projection
     * (fr.c viSetupCurrentPlayerView): the first one per camera */
    if (g_proj_cam >= 0 && !g_persp_scene_set[g_proj_cam]) {
        g_persp_scene[g_proj_cam] = value;
        g_persp_scene_set[g_proj_cam] = true;
    }
}

void interp_on_segment(unsigned seg, uintptr_t base) {
    if (seg != 3 || g_mode != MODE_RECORD) {
        return;
    }
    Frame &c = cur();
    base = norm(base);
    g_seg3 = base;
    g_mv_inst = false;
    if (base == 0) {
        g_cur_inst = -1;
        return;
    }
    auto it = c.inst_by_base.find(base);
    if (it != c.inst_by_base.end()) {
        g_cur_inst = it->second;
        return;
    }
    c.insts.emplace_back();
    Inst &I = c.insts.back();
    I.base = base;
    I.sig = 0;
    I.cam = -1;
    I.prev = -1;
    I.anchor[0] = I.anchor[1] = I.anchor[2] = 0.0f;
    g_cur_inst = (int)c.insts.size() - 1;
    c.inst_by_base[base] = g_cur_inst;
}

bool interp_on_vertex(const void *src, size_t n, const float mv[4][4], const float p[4][4],
                      uint32_t geometry_mode, uintptr_t seg5,
                      float out_mp[4][4], float out_mv[4][4], bool *mv_changed) {
    *mv_changed = false;
    Frame &c = cur();
    if (g_mode == MODE_REPLAY) {
        if (g_rec_ord >= c.recs.size()) {
            g_diverged = true;
            return false;
        }
        /* Replays walk fast3d's recording of the first walk (gfx_pc.cpp), so
         * the source is the recording's copy; the batch sequence is identical
         * by construction and this is only a consistency check. */
        const Rec &r = c.recs[g_rec_ord++];
        if (r.n != (uint32_t)n) {
            if (g_log > 0) {
                sysLogPrintf(LOG_NOTE, "D573 replay diverged at batch %zu/%zu (n %u -> %zu)",
                             g_rec_ord - 1, c.recs.size(), r.n, n);
            }
            g_diverged = true;
            return false;
        }
        return rec_blend(r, out_mp, out_mv, mv_changed);
    }
    const Rec &r = record_vertex(norm((uintptr_t)src), n, mv, p, geometry_mode, norm(seg5));
    return rec_blend(r, out_mp, out_mv, mv_changed);
}

const PortFloatVtx *interp_on_floatvtx(const PortFloatVtx *v, size_t n) {
    const int k = g_proj_cam;
    if (k < 0 || !g_cam_changed[k] || n > sizeof(g_sky_buf) / sizeof(g_sky_buf[0])) {
        return v;
    }
    const float(*Pc)[4] = cur().info.cams[k].P;
    const float(*Pd)[4] = g_Pd[k];
    if (Pc[0][0] == 0.0f || Pc[1][1] == 0.0f || Pc[2][3] == 0.0f) {
        return v;
    }
    const float(*M)[3] = g_sky[k];
    for (size_t i = 0; i < n; i++) {
        PortFloatVtx o = v[i];
        /* clip -> camera space (guPerspective: x' = x P00, y' = y P11, w = z P23) */
        const float x = v[i].x / Pc[0][0];
        const float y = v[i].y / Pc[1][1];
        const float z = v[i].w / Pc[2][3];
        const float x2 = x * M[0][0] + y * M[1][0] + z * M[2][0];
        const float y2 = x * M[0][1] + y * M[1][1] + z * M[2][1];
        const float z2 = x * M[0][2] + y * M[1][2] + z * M[2][2];
        o.x = x2 * Pd[0][0];
        o.y = y2 * Pd[1][1];
        o.w = z2 * Pd[2][3];
        g_sky_buf[i] = o;
    }
    sky_refill_edges(v, g_sky_buf, n);
    return g_sky_buf;
}

void interp_on_scissor(const float vp[4], float *ulx, float *uly, float *lrx, float *lry) {
    const int k = g_proj_cam;
    if (k < 0 || !g_cam_changed[k] || !(vp[2] > 0.0f) || !(vp[3] > 0.0f) || *lrx <= *ulx || *lry <= *uly) {
        return;
    }
    const float e = 1.0f;   /* a side within a pixel of the view's edge is the view's edge */
    const float vx0 = vp[0] - vp[2], vx1 = vp[0] + vp[2], vy0 = vp[1] - vp[3], vy1 = vp[1] + vp[3];
    const bool at_l = *ulx <= vx0 + e, at_r = *lrx >= vx1 - e, at_t = *uly <= vy0 + e, at_b = *lry >= vy1 - e;
    if (at_l && at_r && at_t && at_b) {
        return;   /* the player's whole view (bgScissorCurrentPlayerViewDefault) */
    }
    const float(*Pc)[4] = cur().info.cams[k].P;
    const float(*Pd)[4] = g_Pd[k];
    if (Pc[0][0] == 0.0f || Pc[1][1] == 0.0f || Pc[2][3] == 0.0f) {
        return;
    }
    /* The corners as view directions, turned like the sky (g_sky: this tick's
     * camera space -> the displayed one) and projected again. */
    const float(*M)[3] = g_sky[k];
    float nx0 = 1e9f, nx1 = -1e9f, ny0 = 1e9f, ny1 = -1e9f;
    const float cx[2] = { *ulx, *lrx }, cy[2] = { *uly, *lry };
    bool behind = false;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            const float x = (cx[i] - vp[0]) / vp[2] / Pc[0][0];
            const float y = (vp[1] - cy[j]) / vp[3] / Pc[1][1];
            const float z = 1.0f / Pc[2][3];
            const float x2 = x * M[0][0] + y * M[1][0] + z * M[2][0];
            const float y2 = x * M[0][1] + y * M[1][1] + z * M[2][1];
            const float z2 = x * M[0][2] + y * M[1][2] + z * M[2][2];
            const float w = z2 * Pd[2][3];
            if (!(w > 1e-4f)) {
                behind = true;
                continue;
            }
            const float nx = x2 * Pd[0][0] / w, ny = y2 * Pd[1][1] / w;
            nx0 = fminf(nx0, nx);
            nx1 = fmaxf(nx1, nx);
            ny0 = fminf(ny0, ny);
            ny1 = fmaxf(ny1, ny);
        }
    }
    if (behind) {
        *ulx = vx0;   /* turned past 90 degrees: don't clip */
        *lrx = vx1;
        *uly = vy0;
        *lry = vy1;
        return;
    }
    /* The camera's movement shifts a portal by parallax, which depends on its
     * unknown depth: allow for portals down to kScissorDepth away. */
    float d[3];
    const float(*rot)[3] = cur().cam[k].rot;
    for (int j = 0; j < 3; j++) {
        d[j] = g_cam_rel[k][0] * rot[0][j] + g_cam_rel[k][1] * rot[1][j] + g_cam_rel[k][2] * rot[2][j];
    }
    const float mx = (fabsf(d[0]) * fabsf(Pd[0][0]) + fabsf(d[2])) / kScissorDepth;
    const float my = (fabsf(d[1]) * fabsf(Pd[1][1]) + fabsf(d[2])) / kScissorDepth;
    /* back to screen pixels (y down), sides that were the view's edge stay there */
    *ulx = at_l ? vx0 : fmaxf(vx0, vp[0] + (nx0 - mx) * vp[2]);
    *lrx = at_r ? vx1 : fminf(vx1, vp[0] + (nx1 + mx) * vp[2]);
    *uly = at_t ? vy0 : fmaxf(vy0, vp[1] - (ny1 + my) * vp[3]);
    *lry = at_b ? vy1 : fminf(vy1, vp[1] - (ny0 - my) * vp[3]);
    if (*lrx < *ulx) {
        *lrx = *ulx;
    }
    if (*lry < *uly) {
        *lry = *uly;
    }
}
