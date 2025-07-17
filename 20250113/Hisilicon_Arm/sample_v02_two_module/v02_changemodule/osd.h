#ifndef OSD_SURFACE_H
#define OSD_SURFACE_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "SDL.h"
#include "SDL_ttf.h"
#include <time.h>
#include "sample_comm.h"

#ifndef ALIGN_2
#define ALIGN_2(x) (((x) + 1) & ~1)
#endif

#ifndef MAX_VALUE
#define MAX_VALUE(a, b) ((a) > (b) ? (a) : (b))
#endif

/* 全局变量声明（使用extern） */
extern TTF_Font *font;
extern SDL_Surface *text;
extern SDL_Surface *unicode_surface_temp;

int OpenOsdText();
int CloseOsdText();
void Sample_SurfaceWord_ToBMP(char * pstr,ot_bmp *stBitmap);
int CopyBmpToCanvas(ot_bmp *bmp, ot_rgn_canvas_info *canvas);

#endif