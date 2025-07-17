#include "osd.h"

TTF_Font *font = NULL;
SDL_Surface *text = NULL;
SDL_Surface *unicode_surface_temp = NULL;
SDL_PixelFormat *fmt = NULL;
SDL_Color forecol = {  0x32, 0xcd, 0x32, 0xff };
SDL_Color bg_col = {  0x00, 0x00, 0x00, 0x00 };

int OpenOsdText(){

    if (TTF_Init() < 0 ) 
    {  
        fprintf(stderr, "Couldn't initialize TTF: %s\n",SDL_GetError());  
        SDL_Quit();
    }  

    font = TTF_OpenFont("/root/nfs_share/wqy-microhei.ttc", 48); 
    if ( font == NULL ) 
    {  
        fprintf(stderr, "Couldn't load %d pt font from %s: %s\n", 48, "ptsize", SDL_GetError());  
    }  

    fmt = (SDL_PixelFormat*)malloc(sizeof(SDL_PixelFormat));
    memset(fmt,0,sizeof(SDL_PixelFormat));
    fmt->BitsPerPixel = 16;
    fmt->BytesPerPixel = 2;
    fmt->colorkey = 0xffffffff;
    fmt->alpha = 0xff;

    //SDL_SaveBMP(unicode_surface_temp, "save.bmp"); 
    return 0;
}


int CloseOsdText() {
    SDL_FreeSurface(text);  
    SDL_FreeSurface(unicode_surface_temp);
    TTF_CloseFont(font);  
    TTF_Quit();  
    return 0;
}

void Sample_SurfaceWord_ToBMP(char * pstr,ot_bmp *stBitmap)
{
    // 释放旧的data内存
    if (stBitmap->data != NULL) {
        free(stBitmap->data);
        stBitmap->data = NULL;  // 避免野指针
    }
    if (text != NULL) {
        SDL_FreeSurface(text); 
    }
    if (unicode_surface_temp != NULL) {
        SDL_FreeSurface(unicode_surface_temp);
    }
   
    //text = TTF_RenderUTF8_Solid(font, pstr, forecol);
    
    text = TTF_RenderUTF8(font, pstr, forecol ,bg_col);
    
    //text = TTF_RenderText_Blended_Wrapped(font ,pstr ,forecol ,1000);

    unicode_surface_temp = SDL_ConvertSurface(text, fmt, 0);

	SDL_Color fntcol = {  0xff, 0xff, 0xff, 0xff };  
	td_u16 wrd_color = ((fntcol.r >> 3) << 11) + ((fntcol.g >> 2) << 5) + ((fntcol.b >> 3) << 0); //RGB888=>RGB565
	td_u16 bg_color = 0xffff - wrd_color;  //RGB565下背景色的计算
	stBitmap->height = ALIGN_2(unicode_surface_temp->h); //BITMAP 的宽高向上2对齐
	stBitmap->width = ALIGN_2(unicode_surface_temp->w);
	stBitmap->data = malloc(2*(stBitmap->height)*(stBitmap->width)); //申请空间，RGB1555=>2Byte/Pixel，总大小为2*w*h
	memset(stBitmap->data,0,2*(stBitmap->height)*(stBitmap->width));
	int i,j;
	int w = unicode_surface_temp->w;
	int h = unicode_surface_temp->h;
	for (i = 0; i < h; ++i)
	{
		td_u16 *p_dst = (td_u16*)stBitmap->data;
		td_u16 *p_src = (td_u16*)unicode_surface_temp->pixels;
	    int dis_pos = 0;
	    if(w % 2 != 0)
	    	dis_pos = i;     //处理w为奇数的情况
		for(j=0;j<w;j++)
		{
			int a,r, g , b;
			r = (p_src[i*w+dis_pos+j] & 0xF800) >> 11;   //原图像是RGB565，RGB各分量提取
			g = (p_src[i*w+dis_pos+j] & 0x07e0) >> 5;
			b = (p_src[i*w+dis_pos+j] & 0x001f);
			a = (bg_color==p_src[i*w+dis_pos+j])?0:(1<<15);  //A分量，计算当前颜色和背景色是否一致
            											   //一致则A位设置为0，透明

			p_dst[i*stBitmap->width+j] = (r<<10)+((g>>1)<<5)+b+a;  //转换成RGB1555
			//剩下一个问题，转换为1555后H265输出颜色有点不正常。转换逻辑没问题，应该是内部显示的效果问题
		}
	}
	stBitmap->pixel_format = OT_PIXEL_FORMAT_ARGB_1555;
}

/**
 * 将ot_bmp图像数据bg_color复制到ot_rgn_canvas_info画布
 * @param bmp       源位图数据
 * @param canvas    目标画布
 * @return 0表示成功，非0表示失败
 */
int CopyBmpToCanvas(ot_bmp *bmp, ot_rgn_canvas_info *canvas) {
    /* 错误处理 */
    if (!bmp || !bmp->data) {
        printf("Error: No bmp data.\n");
        return -1;
    }
    
    /* 检查像素格式是否兼容 */
    if (bmp->pixel_format != canvas->pixel_format) {
        printf("Error: Pixel format mismatch! bmp: %d, canvas: %d\n", 
               bmp->pixel_format, canvas->pixel_format);
        return -2;
    }
    
    /* 计算像素字节数 */
    uint32_t bytes_per_pixel = 0;
    switch (bmp->pixel_format) {
        case OT_PIXEL_FORMAT_RGB_888:
            bytes_per_pixel = 3;
            break;
        case OT_PIXEL_FORMAT_RGB_565:
        case OT_PIXEL_FORMAT_ARGB_1555:
            bytes_per_pixel = 2;
            break;
        default:
            printf("Error: Unsupport pixel format: %d\n", bmp->pixel_format);
            return -3;
    }
    
    /* 计算源位图实际占用内存大小（考虑stride） */
    uint32_t src_stride = bmp->width * bytes_per_pixel;
    uint32_t dst_stride = canvas->stride;
    uint32_t height = bmp->height;
    
    /* 检查目标画布是否足够大 */
    uint32_t required_size = dst_stride * height;
    if (required_size > (uint32_t)(canvas->size.width * canvas->size.height)) {  // 类型转换
        printf("Error: Canvas size insufficient! Required: %u, Available: %lu\n", 
               required_size, (unsigned long)(canvas->size.width * canvas->size.height));  // 匹配格式说明符
        return -4;
    }TTF_Font *font = NULL;
    SDL_Surface *text = NULL;
    SDL_Surface *unicode_surface_temp = NULL;
    
    /* 内存复制 - 处理stride差异 */
    uint8_t *src = (uint8_t *)bmp->data;
    uint8_t *dst = (uint8_t *)canvas->virt_addr;
    
    for (uint32_t i = 0; i < height; i++) {
        /* 复制当前行 */
        memcpy(dst + i * dst_stride, src + i * src_stride, src_stride);
        
        /* 如果stride不同，填充剩余空间（可选） */
        if (dst_stride > src_stride) {
            memset(dst + i * dst_stride + src_stride, 0, dst_stride - src_stride);
        }
    }
    return 0;
}