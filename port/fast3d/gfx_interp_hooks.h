#ifndef GFX_INTERP_HOOKS_H
#define GFX_INTERP_HOOKS_H

/* D573: fast3d-internal hooks for frame interpolation (see port/include/gfx_interp.h).
 * Called from gfx_pc.cpp only, on the render thread. All are no-ops (and
 * cheap) unless a recording or replaying pass is in progress. */

#include <stddef.h>
#include <stdint.h>

#include "floatvtx.h"

/* True while a recording or replaying pass is walking the game's display list. */
bool interp_hooks_active(void);

/* gfx_run: 0 = recording pass, 1 = replaying pass, -1 = interpolation off.
 * The pass ends before the F10 overlay is drawn (it is never interpolated). */
int interp_pass_begin(void);
void interp_pass_end(bool drew);

/* G_MTX: `matrix` is the decoded value. On replay it is replaced by the value
 * recorded on the first pass (static matrices may have been rewritten since). */
void interp_on_matrix(uint8_t params, const void *addr, uintptr_t w1, float matrix[4][4]);
/* G_MW_SEGMENT */
void interp_on_segment(unsigned seg, uintptr_t base);
/* G_MW_PERSPNORM: the first-person gun is drawn under its own value
 * (gunfire.c gunRenderFirstPersonGunModels), the scene under the one set
 * with its projection (fr.c viSetupCurrentPlayerView). */
void interp_on_perspnorm(uint32_t value);
/* G_VTX: returns true and fills out_mp (and out_mv when *mv_changed) if this
 * batch must be transformed with blended matrices. */
bool interp_on_vertex(const void *src, size_t n, const float mv[4][4], const float p[4][4],
                      uint32_t geometry_mode, uintptr_t seg5,
                      float out_mp[4][4], float out_mv[4][4], bool *mv_changed);
/* G_FLOATVTX_EXT (sky): returns the vertices to load (possibly a corrected copy). */
const PortFloatVtx *interp_on_floatvtx(const PortFloatVtx *v, size_t n);
/* G_SETSCISSOR, in N64 screen pixels (y down); vp = the current SP viewport's
 * centre x, y and half-size x, y in the same units. A room's clip rectangle
 * (bg.c, the screen box of the portal it is seen through) is moved with the
 * displayed view. */
void interp_on_scissor(const float vp[4], float *ulx, float *uly, float *lrx, float *lry);

#endif
