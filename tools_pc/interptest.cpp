/* D573: standalone numerical test of the frame-interpolation math
 * (port/fast3d/gfx_interp.cpp, included directly): 4x4 inverse, affine
 * decompose/compose round trips (scaled, mirrored), camera slerp against GE's
 * own matrix_4x4_set_lookat, mouse-look yaw/pitch signs against the game's
 * forward vector, the world/camera split, sky re-projection, the sky
 * edge refill (texture coordinates and w taken from the sky plane), static
 * level geometry under a scaled camera with per-tick homogeneous error, and
 * room clip rectangles moved with a turned view.
 *
 * Build and run from the repo root (no game data needed):
 *   g++ -std=c++17 -O1 -I port/fast3d -I port/include -idirafter include \
 *       -DPORT tools_pc/interptest.cpp -o interptest && ./interptest
 * (`-idirafter`: the decomp's include/ shadows some system headers.)
 * Prints "OK (0 failures)" and exits 0 when every check passes.
 */
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <random>

#include "gfx_interp.cpp"

extern "C" uint64_t sysGetMicroseconds(void) { return 0; }
extern "C" void sysLogPrintf(enum LogLevel, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap); printf("\n");
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static float maxdiff(const M4 a, const M4 b) {
    float d = 0; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) d = fmaxf(d, fabsf(a[i][j] - b[i][j])); return d;
}

// GE's matrix_4x4_set_lookat (matrixmath.c), verbatim math
static void ge_lookat(M4 m, float ex, float ey, float ez, float fx, float fy, float fz, float ux, float uy, float uz) {
    float nf = -1.0f / sqrtf(fx * fx + fy * fy + fz * fz); fx *= nf; fy *= nf; fz *= nf;
    float rx = uy * fz - uz * fy, ry = uz * fx - ux * fz, rz = ux * fy - uy * fx;
    float nr = 1.0f / sqrtf(rx * rx + ry * ry + rz * rz); rx *= nr; ry *= nr; rz *= nr;
    ux = fy * rz - fz * ry; uy = fz * rx - fx * rz; uz = fx * ry - fy * rx;
    float nu = 1.0f / sqrtf(ux * ux + uy * uy + uz * uz); ux *= nu; uy *= nu; uz *= nu;
    m[0][0] = rx; m[1][0] = ry; m[2][0] = rz; m[3][0] = -(ex * rx + ey * ry + ez * rz);
    m[0][1] = ux; m[1][1] = uy; m[2][1] = uz; m[3][1] = -(ex * ux + ey * uy + ez * uz);
    m[0][2] = fx; m[1][2] = fy; m[2][2] = fz; m[3][2] = -(ex * fx + ey * fy + ez * fz);
    m[0][3] = m[1][3] = m[2][3] = 0; m[3][3] = 1;
}
static void ge_cam(M4 V, float yaw_deg, float pitch_deg, float ex, float ey, float ez) {
    const float d = 3.14159265f / 180.0f;
    float th = yaw_deg * d, ph = pitch_deg * d;
    ge_lookat(V, ex, ey, ez, -sinf(th) * cosf(ph), sinf(ph), cosf(th) * cosf(ph), 0, 1, 0);
}
static void persp(M4 P, float fovy_deg, float aspect, float n, float f) {
    memset(P, 0, sizeof(M4));
    float c = 1.0f / tanf(fovy_deg * 3.14159265f / 360.0f);
    P[0][0] = c / aspect; P[1][1] = c; P[2][2] = (n + f) / (n - f); P[2][3] = -1; P[3][2] = 2 * n * f / (n - f);
}

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1, 1);

    // 1. inverse
    for (int t = 0; t < 200; t++) {
        M4 m, inv, prod, I = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) m[i][j] = U(rng) * 3;
        if (!m4_invert(inv, m)) continue;
        m4_mul(prod, m, inv);
        CHECK(maxdiff(prod, I) < 1e-3f, "inverse %g", maxdiff(prod, I));
    }

    // 2. decompose/compose round trip incl. mirrored and scaled
    for (int t = 0; t < 200; t++) {
        float q[4] = {U(rng), U(rng), U(rng), U(rng)}; float n = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
        for (float &x : q) x /= n;
        float r[3][3]; m3_from_quat(q, r);
        float s[3] = {0.5f + fabsf(U(rng)) * 3, 0.5f + fabsf(U(rng)) * 3, 0.5f + fabsf(U(rng)) * 3};
        if (t % 3 == 0) s[1] = -s[1];
        M4 m; for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) m[i][j] = r[i][j] * s[i]; m[i][3] = 0; m[3][i] = U(rng) * 5000; } m[3][3] = 1;
        Aff a; bool ok = aff_decompose(m, a);
        CHECK(ok, "decompose failed");
        M4 m2; aff_compose(m2, a);
        CHECK(maxdiff(m, m2) < 2e-3f, "roundtrip %g", maxdiff(m, m2));
    }

    // 3. blend at t=0 and t=1 reproduces the ends; midpoint of a pure yaw is the mid yaw
    {
        M4 A, B, Mid, Exp; ge_cam(A, 10, 5, 100, 50, -300); ge_cam(B, 40, 5, 100, 50, -300); ge_cam(Exp, 25, 5, 100, 50, -300);
        Aff a, b, o; aff_decompose(A, a); aff_decompose(B, b);
        aff_blend(o, a, b, 0); M4 r0; aff_compose(r0, o); CHECK(maxdiff(r0, A) < 2e-3f, "t0");
        aff_blend(o, a, b, 1); M4 r1; aff_compose(r1, o); CHECK(maxdiff(r1, B) < 2e-3f, "t1");
        // camera blend as setup_pass_cams does it: slerp rotation, lerp position
        GfxInterpCam ga = {}, gb = {}; ga.valid = gb.valid = 1; m4_copy(ga.V, A); m4_copy(gb.V, B);
        CamInfo ca, cb; M4 ia, ib, za, zb; cam_info_from(ga, ca, ia, za); cam_info_from(gb, cb, ib, zb);
        float q[4]; q_slerp(q, ca.a.q, cb.a.q, 0.5f); float R[3][3]; m3_from_quat(q, R);
        for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) Mid[i][j] = R[i][j]; Mid[i][3] = 0; }
        for (int j = 0; j < 3; j++) Mid[3][j] = -(ca.pos[0]*Mid[0][j] + ca.pos[1]*Mid[1][j] + ca.pos[2]*Mid[2][j]); Mid[3][3] = 1;
        // pitch fixed at 5 while yaw moves: slerp keeps the camera nearly level (roll ~0)
        CHECK(maxdiff(Mid, Exp) < 5e-3f, "camera mid yaw %g", maxdiff(Mid, Exp));
        CHECK(fabsf(ca.pos[0] - 100) < 1e-2f && fabsf(ca.pos[2] + 300) < 1e-2f, "camera pos %g %g %g", ca.pos[0], ca.pos[1], ca.pos[2]);
    }

    // 4. mouse look: camera_apply_look(yaw, pitch) == the game's camera at yaw+dy, pitch+dp
    for (int t = 0; t < 50; t++) {
        float y0 = U(rng) * 180, p0 = U(rng) * 60, dy = U(rng) * 20, dp = U(rng) * 20;
        M4 A, B; ge_cam(A, y0, p0, 10, 20, 30); ge_cam(B, y0 + dy, p0 + dp, 10, 20, 30);
        float R[3][3]; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R[i][j] = A[i][j];
        camera_apply_look(R, dy, dp);
        float e = 0; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) e = fmaxf(e, fabsf(R[i][j] - B[i][j]));
        CHECK(e < 2e-3f, "look y0=%g p0=%g dy=%g dp=%g err=%g", y0, p0, dy, dp, e);
        // heading formula used by the GE_INTERP_LOG yaw check
        float h = atan2f(A[0][2], -A[2][2]) * 180 / 3.14159265f;
        float dh = h - y0; while (dh > 180) dh -= 360; while (dh < -180) dh += 360;
        CHECK(fabsf(dh) < 1e-2f, "heading %g vs %g", h, y0);
    }

    // 5. sky: a direction projected with V0*P, corrected to V1*P, equals projecting with V1*P
    {
        M4 P; persp(P, 60, 1.333f, 10, 10000);
        M4 V0, V1; ge_cam(V0, 30, 10, 0, 0, 0); ge_cam(V1, 37, 4, 0, 0, 0);
        GfxInterpCam g0 = {}, g1 = {}; g0.valid = g1.valid = 1; m4_copy(g0.V, V0); m4_copy(g1.V, V1);
        CamInfo c0, c1; M4 i0, i1, z0, z1; cam_info_from(g0, c0, i0, z0); cam_info_from(g1, c1, i1, z1);
        float Rt[3][3], M[3][3]; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rt[i][j] = c0.rot[j][i];
        m3_mul(M, Rt, c1.rot);
        for (int t = 0; t < 20; t++) {
            float d[4] = {U(rng) * 1000, U(rng) * 300, U(rng) * 1000, 1};
            float v0[4], v1[4], cl0[4], cl1[4];
            for (int j = 0; j < 4; j++) { v0[j] = 0; v1[j] = 0; for (int i = 0; i < 4; i++) { v0[j] += d[i] * V0[i][j]; v1[j] += d[i] * V1[i][j]; } }
            for (int j = 0; j < 4; j++) { cl0[j] = 0; cl1[j] = 0; for (int i = 0; i < 4; i++) { cl0[j] += v0[i] * P[i][j]; cl1[j] += v1[i] * P[i][j]; } }
            float x = cl0[0] / P[0][0], y = cl0[1] / P[1][1], z = cl0[3] / P[2][3];
            float x2 = x * M[0][0] + y * M[1][0] + z * M[2][0];
            float y2 = x * M[0][1] + y * M[1][1] + z * M[2][1];
            float z2 = x * M[0][2] + y * M[1][2] + z * M[2][2];
            float ox = x2 * P[0][0], oy = y2 * P[1][1], ow = z2 * P[2][3];
            CHECK(fabsf(ox - cl1[0]) < 1e-1f && fabsf(oy - cl1[1]) < 1e-1f && fabsf(ow - cl1[3]) < 1e-1f,
                  "sky (%g %g %g) vs (%g %g %g)", ox, oy, ow, cl1[0], cl1[1], cl1[3]);
        }
    }

    // 6. world/camera split: W = C * V^-1, C' = W * V
    {
        M4 V; ge_cam(V, 70, -10, 500, 100, 900);
        M4 Wm; Aff a; float q[4] = {0.2f, 0.3f, 0.1f, 0.9f}; float n = sqrtf(0.04f+0.09f+0.01f+0.81f); for (float &x : q) x /= n;
        memcpy(a.q, q, sizeof q); a.s[0] = a.s[1] = a.s[2] = 1.5f; a.t[0] = 40; a.t[1] = 5; a.t[2] = 700; aff_compose(Wm, a);
        M4 C, Vi, W2, C2; m4_mul(C, Wm, V); m4_invert(Vi, V); m4_mul(W2, C, Vi); m4_mul(C2, W2, V);
        CHECK(maxdiff(W2, Wm) < 1e-2f, "W split %g", maxdiff(W2, Wm));
        CHECK(maxdiff(C2, C) < 1e-2f, "C rebuild %g", maxdiff(C2, C));
    }

    // 7. sky edge refill: a cloud plane y = H with s = x/10, t = z/10, whose fan is
    //    the screen rectangle under V1 (right and bottom a quarter pixel short, as
    //    skyClamp leaves them); turned to V0, left and then right, the corners must
    //    come back to the corners with the s, t, w of the plane point V0 sees there.
    for (int dir = 0; dir < 2; dir++) {
        M4 P; persp(P, 60, 1.333f, 10, 100000);
        M4 V0, V1; ge_cam(V0, dir ? 36 : 30, 35, 0, 0, 0); ge_cam(V1, dir ? 30 : 36, 31, 0, 0, 0);
        const float H = 3000;
        auto corner = [&](const M4 V, float nx, float ny, PortFloatVtx &out) {
            // camera-space ray through NDC (nx, ny): x = nx*w/P00, y = ny*w/P11, z = -w
            M4 Vi; m4_invert(Vi, V);
            float dc[3] = {nx / P[0][0], ny / P[1][1], -1.0f};   // per unit w
            float dw[3], o[3];
            for (int j = 0; j < 3; j++) { dw[j] = dc[0]*Vi[0][j] + dc[1]*Vi[1][j] + dc[2]*Vi[2][j]; o[j] = Vi[3][j]; }
            float w = (H - o[1]) / dw[1];
            float px = o[0] + dw[0]*w, pz = o[2] + dw[2]*w;
            out.x = nx * w; out.y = ny * w; out.z = 0; out.w = w; out.s = px / 10; out.t = pz / 10;
        };
        PortFloatVtx fan[4], exp0[4];
        float cs[4][2] = {{-1,-0.9977f},{0.9984f,-0.9977f},{0.9984f,1},{-1,1}};
        for (int i = 0; i < 4; i++) { corner(V1, cs[i][0], cs[i][1], fan[i]); corner(V0, cs[i][0], cs[i][1], exp0[i]); }
        // rotate as interp_on_floatvtx does (camera at origin: rotation only)
        GfxInterpCam g0 = {}, g1 = {}; g0.valid = g1.valid = 1; m4_copy(g0.V, V0); m4_copy(g1.V, V1);
        CamInfo c0, c1; M4 i0, i1, z0, z1; cam_info_from(g0, c0, i0, z0); cam_info_from(g1, c1, i1, z1);
        float Rt[3][3], M[3][3]; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rt[i][j] = c1.rot[j][i];
        m3_mul(M, Rt, c0.rot);
        PortFloatVtx o[4];
        for (int i = 0; i < 4; i++) {
            o[i] = fan[i];
            float x = fan[i].x / P[0][0], y = fan[i].y / P[1][1], z = fan[i].w / P[2][3];
            o[i].x = (x*M[0][0] + y*M[1][0] + z*M[2][0]) * P[0][0];
            o[i].y = (x*M[0][1] + y*M[1][1] + z*M[2][1]) * P[1][1];
            o[i].w = (x*M[0][2] + y*M[1][2] + z*M[2][2]) * P[2][3];
        }
        sky_refill_edges(fan, o, 4);
        int pinned = 0;
        for (int i = 0; i < 4; i++) {
            const float nx = o[i].x / o[i].w, ny = o[i].y / o[i].w;
            // coverage: the vertex is at or beyond its screen corner on both axes
            CHECK(nx * (cs[i][0] > 0 ? 1 : -1) >= 0.997f && ny * (cs[i][1] > 0 ? 1 : -1) >= 0.997f,
                  "refill dir %d corner %d inside the screen at (%g %g)", dir, i, nx, ny);
            if (fabsf(fabsf(nx) - 1) < 1e-4f || fabsf(fabsf(ny) - 1) < 1e-4f) pinned++;
            // consistency: s, t, w are the plane's at the vertex's own NDC under V0
            PortFloatVtx e; corner(V0, nx, ny, e);
            CHECK(fabsf(o[i].w - e.w) < 2e-3f * e.w, "refill w %d %g vs %g", i, o[i].w, e.w);
            CHECK(fabsf(o[i].s - e.s) < 1e-3f * fabsf(e.s) + 0.5f && fabsf(o[i].t - e.t) < 1e-3f * fabsf(e.t) + 0.5f,
                  "refill st %d (%g %g) vs (%g %g)", i, o[i].s, o[i].t, e.s, e.t);
        }
        CHECK(pinned > 0, "nothing was refilled (test does not exercise the edge)");
    }

    // 8. static level geometry far from the world origin (Dam), seen by a camera
    //    scaled by the stage's visibility scale (0.2), through a C that carries a
    //    different homogeneous scale each tick (fixed-point projection x view over
    //    the fixed-point projection): blended between the two ticks and drawn
    //    from the blended camera, it must land where that camera sees it.
    {
        const float s = 0.2f;
        M4 P; persp(P, 60, 1.333f, 2, 20000);
        auto scaled_cam = [&](M4 V, float yaw, float ex, float ey, float ez) {
            ge_cam(V, yaw, -2, ex, ey, ez);
            for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) V[i][j] *= s;
        };
        const float cp[3] = {15600, 760, 19715}, cc[3] = {15612, 756, 19716};
        M4 Vp, Vc; scaled_cam(Vp, 224, cp[0], cp[1], cp[2]); scaled_cam(Vc, 225, cc[0], cc[1], cc[2]);
        GfxInterpCam gp = {}, gc = {}; gp.valid = gc.valid = 1; m4_copy(gp.V, Vp); m4_copy(gc.V, Vc);
        CamInfo ip, ic; M4 vip, vic, v0p, v0c;
        CHECK(cam_info_from(gp, ip, vip, v0p) && cam_info_from(gc, ic, vic, v0c), "scaled camera rejected");
        CHECK(fabsf(ic.pos[0] - cc[0]) < 0.05f && fabsf(ic.pos[2] - cc[2]) < 0.05f, "scaled camera pos %g %g", ic.pos[0], ic.pos[2]);
        // a room at (18900, 757, 16517), 1.2 world units per bg unit
        M4 Mw = {{1.2f,0,0,0},{0,1.2f,0,0},{0,0,1.2f,0},{18900,757,16517,1}};
        Rec rp = {}, rc = {};
        m4_mul(rp.C, Mw, Vp); m4_mul(rc.C, Mw, Vc);
        // w errors inside aff_decompose's tolerance: dropped silently if not divided out
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { rp.C[i][j] *= 1.0008f; rc.C[i][j] *= 0.9993f; }
        scene_world(rp.C, rp.W, rp.pos, v0p, ip.pos); scene_world(rc.C, rc.W, rc.pos, v0c, ic.pos);
        rp.aff_ok = aff_decompose(rp.W, rp.aff); rc.aff_ok = aff_decompose(rc.W, rc.aff);
        CHECK(rp.aff_ok && rc.aff_ok, "room W not affine");
        CHECK(dist3(rp.pos, rc.pos) < 0.05f, "static room moved %g", dist3(rp.pos, rc.pos));
        for (float a : {0.0f, 0.25f, 0.5f, 0.95f}) {
            // the displayed camera as setup_pass_cams builds it, relative to this tick's camera
            float q[4], R[3][3], pos[3], rel[3]; q_slerp(q, ip.a.q, ic.a.q, a); m3_from_quat(q, R);
            M4 Vd, Vr;
            for (int j = 0; j < 3; j++) { pos[j] = ip.pos[j] + (ic.pos[j] - ip.pos[j]) * a; rel[j] = pos[j] - ic.pos[j]; }
            for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) Vd[i][j] = Vr[i][j] = R[i][j] * s; Vd[i][3] = Vr[i][3] = 0; }
            for (int j = 0; j < 3; j++) {
                Vd[3][j] = -(pos[0]*Vd[0][j] + pos[1]*Vd[1][j] + pos[2]*Vd[2][j]);
                Vr[3][j] = -(rel[0]*Vd[0][j] + rel[1]*Vd[1][j] + rel[2]*Vd[2][j]);
            }
            Vd[3][3] = Vr[3][3] = 1;
            const float poff[3] = {ip.pos[0] - ic.pos[0], ip.pos[1] - ic.pos[1], ip.pos[2] - ic.pos[2]};
            M4 W, MP, VPr, ideal, VPd; blend_pair(W, rp, rc, true, a, poff);
            m4_mul(VPr, Vr, P); m4_mul(MP, W, VPr);
            m4_mul(VPd, Vd, P); m4_mul(ideal, Mw, VPd);
            const float v[4] = {400, 30, -250, 1};   // a vertex 480 world units from the room origin
            float x0[4], x1[4];
            for (int j = 0; j < 4; j++) { x0[j] = x1[j] = 0; for (int i = 0; i < 4; i++) { x0[j] += v[i]*MP[i][j]; x1[j] += v[i]*ideal[i][j]; } }
            const float px = 540 * hypotf(x0[0]/x0[3] - x1[0]/x1[3], x0[1]/x0[3] - x1[1]/x1[3]);
            CHECK(x0[3] > 0 && x1[3] > 0 && px < 0.05f, "static room at a=%g off by %g px (w %g %g)", a, px, x0[3], x1[3]);
        }
    }

    // 7b. the sky fan exactly as sky.c drew it on Dam (captured with a 1333 deg/s
    //     synthetic mouse): its lower corners are the cloud quad's far cut just
    //     above the plane's horizon, clamped to the screen sides with the w (and
    //     folded s, t) of the unclamped points, so the four are only roughly
    //     coplanar. Turned right by the two captured rotations, then 10-48
    //     degrees about the same axis with -3..3 degrees of pitch (the predicted
    //     view's lead in a hard whip), the right-hand corners must still reach
    //     the right edge, in front. (The fitted plane puts some of these edge
    //     points past its horizon with any downward pitch; they used to be
    //     skipped, leaving the corner mid-screen.)
    {
        const PortFloatVtx fan[4] = {
            {-9758.6084f, 9758.6084f, 0, 9758.6084f, 23519.875f, 21581.8828f, 0, 0, 0, 0},
            {9743.3623f, 9758.61035f, 0, 9758.61035f, 23544.2129f, 19941.0176f, 0, 0, 0, 0},
            {-229257.438f, 27822.2051f, 0, 229257.438f, 1279.65625f, 39715.8594f, 0, 0, 0, 0},
            {228958.297f, 27829.3711f, 0, 229316.594f, 1845.58008f, 1161.87695f, 0, 0, 0, 0},
        };
        const float P00 = 1.18891907f, P11 = 1.73204041f, P23 = -1.0f;
        const float Ms[2][3][3] = {
            {{0.897459507f, 0.0309869107f, -0.440007061f}, {-0.030767614f, 0.999497414f, 0.00763316592f}, {0.440022439f, 0.00668751774f, 0.897961915f}},
            {{0.795691252f, 0.0423679166f, -0.604219079f}, {-0.0422492921f, 0.999003172f, 0.0144124757f}, {0.604227304f, 0.0140599422f, 0.796687901f}},
        };
        float k[3] = {Ms[1][2][1] - Ms[1][1][2], Ms[1][0][2] - Ms[1][2][0], Ms[1][1][0] - Ms[1][0][1]};
        const float kn = sqrtf(k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
        for (float &x : k) x /= kn;
        int failed = 0;
        for (int m = 0; m < 2 + 39 * 7; m++) {
            float M[3][3];
            if (m < 2) {
                memcpy(M, Ms[m], sizeof M);
            } else {
                const float th = (10 + (m - 2) / 7) * 3.14159265f / 180, c = cosf(th), sn = sinf(th), t = 1 - c;
                const float Y[3][3] = {{c + k[0]*k[0]*t, k[0]*k[1]*t - k[2]*sn, k[0]*k[2]*t + k[1]*sn},
                                       {k[1]*k[0]*t + k[2]*sn, c + k[1]*k[1]*t, k[1]*k[2]*t - k[0]*sn},
                                       {k[2]*k[0]*t - k[1]*sn, k[2]*k[1]*t + k[0]*sn, c + k[2]*k[2]*t}};
                const float g = ((m - 2) % 7 - 3) * 3.14159265f / 180, cg = cosf(g), sg = sinf(g);
                const float X[3][3] = {{1, 0, 0}, {0, cg, sg}, {0, -sg, cg}};
                m3_mul(M, X, Y);
            }
            PortFloatVtx o[4];
            for (int i = 0; i < 4; i++) {
                o[i] = fan[i];
                const float x = fan[i].x / P00, y = fan[i].y / P11, z = fan[i].w / P23;
                o[i].x = (x * M[0][0] + y * M[1][0] + z * M[2][0]) * P00;
                o[i].y = (x * M[0][1] + y * M[1][1] + z * M[2][1]) * P11;
                o[i].w = (x * M[0][2] + y * M[1][2] + z * M[2][2]) * P23;
            }
            sky_refill_edges(fan, o, 4);
            for (int i : {1, 3}) {
                const float nx = o[i].w > 0 ? o[i].x / o[i].w : -9;
                if (!(o[i].w > 0 && nx >= 0.997f) && failed++ < 3) {
                    CHECK(false, "captured fan, turn %d: corner %d not at the right edge (%g, w %g)", m, i, nx, o[i].w);
                }
            }
        }
        CHECK(failed == 0, "captured fan: %d corners short of the right edge", failed);
    }

    // 9. room clip rectangle (G_SETSCISSOR): a portal quad's screen box under the
    //    tick's camera V1, moved by interp_on_scissor to the displayed V0 (turned
    //    6 degrees, same position), must be the quad's box under V0; a side on the
    //    view's edge must stay there; the whole view must be left alone.
    {
        M4 P; persp(P, 60, 1.333f, 10, 10000);
        M4 V0, V1; ge_cam(V0, 36, 3, 0, 0, 0); ge_cam(V1, 30, 1, 0, 0, 0);
        const float vp[4] = {160, 120, 160, 110};   // N64 viewport: centre, half size
        auto screen = [&](const M4 V, const float p[3], float &sx, float &sy) {
            float c[4] = {0, 0, 0, 0}, h[4] = {p[0], p[1], p[2], 1}, cl[4] = {0, 0, 0, 0};
            for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) c[j] += h[i] * V[i][j];
            for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) cl[j] += c[i] * P[i][j];
            sx = vp[0] + cl[0] / cl[3] * vp[2]; sy = vp[1] - cl[1] / cl[3] * vp[3];
        };
        auto box = [&](const M4 V, float b[4]) {   // a 120 x 80 doorway 500 ahead of V1's view
            const float q[4][3] = {{-300, 20, 400}, {-180, 20, 400}, {-180, 100, 400}, {-300, 100, 400}};
            b[0] = b[1] = 1e9f; b[2] = b[3] = -1e9f;
            for (auto &p : q) { float x, y; screen(V, p, x, y); b[0] = fminf(b[0], x); b[1] = fminf(b[1], y); b[2] = fmaxf(b[2], x); b[3] = fmaxf(b[3], y); }
        };
        GfxInterpCam g0 = {}, g1 = {}; g0.valid = g1.valid = 1; m4_copy(g0.V, V0); m4_copy(g1.V, V1);
        CamInfo c0, c1; M4 i0, i1, z0, z1; cam_info_from(g0, c0, i0, z0); cam_info_from(g1, c1, i1, z1);
        // the pass state setup_pass_cams leaves for a turned view
        g_cur = 0; g_fr[0].cam[0] = c1; m4_copy(g_fr[0].info.cams[0].P, P);
        g_proj_cam = 0; g_cam_changed[0] = true; m4_copy(g_Pd[0], P);
        g_cam_rel[0][0] = g_cam_rel[0][1] = g_cam_rel[0][2] = 0;
        float Rt[3][3]; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rt[i][j] = c1.rot[j][i];
        m3_mul(g_sky[0], Rt, c0.rot);
        float b1[4], b0[4]; box(V1, b1); box(V0, b0);
        CHECK(b1[0] > 0 && b1[2] < 320 && b0[0] > 0 && b0[2] < 320, "doorway off screen (%g..%g, %g..%g)", b1[0], b1[2], b0[0], b0[2]);
        float r[4] = {b1[0], b1[1], b1[2], b1[3]};
        interp_on_scissor(vp, &r[0], &r[1], &r[2], &r[3]);
        float err = 0; for (int i = 0; i < 4; i++) err = fmaxf(err, fabsf(r[i] - b0[i]));
        CHECK(err < 0.05f, "moved clip (%g %g %g %g) vs (%g %g %g %g)", r[0], r[1], r[2], r[3], b0[0], b0[1], b0[2], b0[3]);
        CHECK(fabsf(b1[0] - b0[0]) > 10, "test turn too small to matter (%g)", b1[0] - b0[0]);
        float e2[4] = {0, b1[1], b1[2], b1[3]};   // left side on the view's edge
        interp_on_scissor(vp, &e2[0], &e2[1], &e2[2], &e2[3]);
        CHECK(e2[0] == 0 && fabsf(e2[2] - b0[2]) < 0.05f, "edge side moved (%g, right %g vs %g)", e2[0], e2[2], b0[2]);
        float f[4] = {0, 10, 320, 230};   // the whole view
        interp_on_scissor(vp, &f[0], &f[1], &f[2], &f[3]);
        CHECK(f[0] == 0 && f[1] == 10 && f[2] == 320 && f[3] == 230, "whole-view clip changed");
        g_proj_cam = -1; g_cam_changed[0] = false;
    }

    printf("%s (%d failures)\n", fails ? "FAILED" : "OK", fails);
    return fails ? 1 : 0;
}
