/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the plugins of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 3 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL3 included in the
** packaging of this file. Please review the following information to
** ensure the GNU Lesser General Public License version 3 requirements
** will be met: https://www.gnu.org/licenses/lgpl-3.0.html.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 2.0 or (at your option) the GNU General
** Public license version 3 or any later version approved by the KDE Free
** Qt Foundation. The licenses are as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL2 and LICENSE.GPL3
** included in the packaging of this file. Please review the following
** information to ensure the GNU General Public License requirements will
** be met: https://www.gnu.org/licenses/gpl-2.0.html and
** https://www.gnu.org/licenses/gpl-3.0.html.
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "qlinuxfbscreen.h"
#include <QtFbSupport/private/qfbcursor_p.h>
#include <QtFbSupport/private/qfbwindow_p.h>
#include <QtCore/QFile>
#include <QtCore/QRegularExpression>
#include <QtGui/QPainter>

#include <private/qcore_unix_p.h> // overrides QT_OPEN
#include <qimage.h>
#include <qdebug.h>

#include <unistd.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <linux/kd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <limits.h>
#include <signal.h>

#include <linux/fb.h>

#include "gfbg.h"
#include "ot_common_tde.h"
#include "ss_mpi_tde.h"
#include "ot_common.h"
#include "ot_math.h"
#include "ot_buffer.h"
#include "ot_defines.h"
#include "securec.h"
#include "ss_mpi_sys.h"
#include "ss_mpi_sys_mem.h"
#include "ss_mpi_vb.h"
#include "ss_mpi_vo.h"
#include "ss_mpi_vo_dev.h"
#define FHD_WIDTH           1280
#define FHD_HEIGHT          1024

QT_BEGIN_NAMESPACE

static int openFramebufferDevice(const QString &dev)
{
    int fd = -1;

    if (access(dev.toLatin1().constData(), R_OK|W_OK) == 0)
        fd = QT_OPEN(dev.toLatin1().constData(), O_RDWR);

    if (fd == -1) {
        if (access(dev.toLatin1().constData(), R_OK) == 0)
            fd = QT_OPEN(dev.toLatin1().constData(), O_RDONLY);
    }

    return fd;
}

static int determineDepth(const fb_var_screeninfo &vinfo)
{
    int depth = vinfo.bits_per_pixel;
    if (depth== 24) {
        depth = vinfo.red.length + vinfo.green.length + vinfo.blue.length;
        if (depth <= 0)
            depth = 24; // reset if color component lengths are not reported
    } else if (depth == 16) {
        depth = vinfo.red.length + vinfo.green.length + vinfo.blue.length;
        if (depth <= 0)
            depth = 16;
    }
    return depth;
}

static QRect determineGeometry(const fb_var_screeninfo &vinfo, const QRect &userGeometry)
{
    int xoff = vinfo.xoffset;
    int yoff = vinfo.yoffset;
    int w, h;
    if (userGeometry.isValid()) {
        w = userGeometry.width();
        h = userGeometry.height();
        if ((uint)w > vinfo.xres)
            w = vinfo.xres;
        if ((uint)h > vinfo.yres)
            h = vinfo.yres;

        int xxoff = userGeometry.x(), yyoff = userGeometry.y();
        if (xxoff != 0 || yyoff != 0) {
            if (xxoff < 0 || xxoff + w > (int)(vinfo.xres))
                xxoff = vinfo.xres - w;
            if (yyoff < 0 || yyoff + h > (int)(vinfo.yres))
                yyoff = vinfo.yres - h;
            xoff += xxoff;
            yoff += yyoff;
        } else {
            xoff += (vinfo.xres - w)/2;
            yoff += (vinfo.yres - h)/2;
        }
    } else {
        w = vinfo.xres;
        h = vinfo.yres;
    }

    if (w == 0 || h == 0) {
        qWarning("Unable to find screen geometry, using 320x240");
        w = 320;
        h = 240;
    }

    return QRect(xoff, yoff, w, h);
}

static QSizeF determinePhysicalSize(const fb_var_screeninfo &vinfo, const QSize &mmSize, const QSize &res)
{
    int mmWidth = mmSize.width(), mmHeight = mmSize.height();

    if (mmWidth <= 0 && mmHeight <= 0) {
        if (vinfo.width != 0 && vinfo.height != 0
            && vinfo.width != UINT_MAX && vinfo.height != UINT_MAX) {
            mmWidth = vinfo.width;
            mmHeight = vinfo.height;
        } else {
            const int dpi = 100;
            mmWidth = qRound(res.width() * 25.4 / dpi);
            mmHeight = qRound(res.height() * 25.4 / dpi);
        }
    } else if (mmWidth > 0 && mmHeight <= 0) {
        mmHeight = res.height() * mmWidth/res.width();
    } else if (mmHeight > 0 && mmWidth <= 0) {
        mmWidth = res.width() * mmHeight/res.height();
    }

    return QSize(mmWidth, mmHeight);
}

static QImage::Format determineFormat(const fb_var_screeninfo &info, int depth)
{
    const fb_bitfield rgba[4] = { info.red, info.green,
                                  info.blue, info.transp };

    QImage::Format format = QImage::Format_Invalid;

    switch (depth) {
    case 32: {
        const fb_bitfield argb8888[4] = {{16, 8, 0}, {8, 8, 0},
                                         {0, 8, 0}, {24, 8, 0}};
        const fb_bitfield abgr8888[4] = {{0, 8, 0}, {8, 8, 0},
                                         {16, 8, 0}, {24, 8, 0}};
        if (memcmp(rgba, argb8888, 4 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_ARGB32;
        } else if (memcmp(rgba, argb8888, 3 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_RGB32;
        } else if (memcmp(rgba, abgr8888, 3 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_RGB32;
            // pixeltype = BGRPixel;
        }
        break;
    }
    case 24: {
        const fb_bitfield rgb888[4] = {{16, 8, 0}, {8, 8, 0},
                                       {0, 8, 0}, {0, 0, 0}};
        const fb_bitfield bgr888[4] = {{0, 8, 0}, {8, 8, 0},
                                       {16, 8, 0}, {0, 0, 0}};
        if (memcmp(rgba, rgb888, 3 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_RGB888;
        } else if (memcmp(rgba, bgr888, 3 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_BGR888;
            // pixeltype = BGRPixel;
        }
        break;
    }
    case 18: {
        const fb_bitfield rgb666[4] = {{12, 6, 0}, {6, 6, 0},
                                       {0, 6, 0}, {0, 0, 0}};
        if (memcmp(rgba, rgb666, 3 * sizeof(fb_bitfield)) == 0)
            format = QImage::Format_RGB666;
        break;
    }
    case 16: {
        const fb_bitfield rgb565[4] = {{11, 5, 0}, {5, 6, 0},
                                       {0, 5, 0}, {0, 0, 0}};
        const fb_bitfield bgr565[4] = {{0, 5, 0}, {5, 6, 0},
                                       {11, 5, 0}, {0, 0, 0}};
        if (memcmp(rgba, rgb565, 3 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_RGB16;
        } else if (memcmp(rgba, bgr565, 3 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_RGB16;
            // pixeltype = BGRPixel;
        }
        break;
    }
    case 15: {
        const fb_bitfield rgb1555[4] = {{10, 5, 0}, {5, 5, 0},
                                        {0, 5, 0}, {15, 1, 0}};
        const fb_bitfield bgr1555[4] = {{0, 5, 0}, {5, 5, 0},
                                        {10, 5, 0}, {15, 1, 0}};
        if (memcmp(rgba, rgb1555, 3 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_RGB555;
        } else if (memcmp(rgba, bgr1555, 3 * sizeof(fb_bitfield)) == 0) {
            format = QImage::Format_RGB555;
            // pixeltype = BGRPixel;
        }
        break;
    }
    case 12: {
        const fb_bitfield rgb444[4] = {{8, 4, 0}, {4, 4, 0},
                                       {0, 4, 0}, {0, 0, 0}};
        if (memcmp(rgba, rgb444, 3 * sizeof(fb_bitfield)) == 0)
            format = QImage::Format_RGB444;
        break;
    }
    case 8:
        break;
    case 1:
        format = QImage::Format_Mono; //###: LSB???
        break;
    default:
        break;
    }

    return format;
}

static int openTtyDevice(const QString &device)
{
    const char *const devs[] = { "/dev/tty0", "/dev/tty", "/dev/console", 0 };

    int fd = -1;
    if (device.isEmpty()) {
        for (const char * const *dev = devs; *dev; ++dev) {
            fd = QT_OPEN(*dev, O_RDWR);
            if (fd != -1)
                break;
        }
    } else {
        fd = QT_OPEN(QFile::encodeName(device).constData(), O_RDWR);
    }

    return fd;
}

static void switchToGraphicsMode(int ttyfd, bool doSwitch, int *oldMode)
{
    // Do not warn if the switch fails: the ioctl fails when launching from a
    // remote console and there is nothing we can do about it.  The matching
    // call in resetTty should at least fail then, too, so we do no harm.
    if (ioctl(ttyfd, KDGETMODE, oldMode) == 0) {
        if (doSwitch && *oldMode != KD_GRAPHICS)
            ioctl(ttyfd, KDSETMODE, KD_GRAPHICS);
    }
}

static void resetTty(int ttyfd, int oldMode)
{
    ioctl(ttyfd, KDSETMODE, oldMode);

    QT_CLOSE(ttyfd);
}

static void blankScreen(int fd, bool on)
{
    ioctl(fd, FBIOBLANK, on ? VESA_POWERDOWN : VESA_NO_BLANKING);
}

QLinuxFbScreen::QLinuxFbScreen(const QStringList &args)
    : mArgs(args), mFbFd(-1), mTtyFd(-1), mBlitter(0)
{
    mMmap.data = 0;
}

static td_bool g_tde_support = TD_TRUE;
static ot_fb_buf g_canvas_buf;
static td_phys_addr_t g_canvas_addr;
static td_void* g_buf_addr = NULL;
static ot_size g_fb_size;
static ot_fb_color_format g_color_format = OT_FB_FORMAT_ARGB8888;	//OT_FB_FORMAT_ARGB1555 OT_FB_FORMAT_ARGB8888


QLinuxFbScreen::~QLinuxFbScreen()
{
    td_bool bShow;

    if (mFbFd != -1) {
        #if 0
        if (mMmap.data){
            munmap(mMmap.data - mMmap.offset, mMmap.size);
        }
        #else
        ss_mpi_sys_mmz_free(g_canvas_addr, g_buf_addr);
        g_canvas_addr = 0;

        bShow = TD_FALSE;
        if (ioctl(mFbFd, FBIOPUT_SHOW_GFBG, &bShow) < 0) {
            qErrnoWarning(errno, "FBIOPUT_SHOW_GFBG failed!");
        }
        #endif
            
        close(mFbFd);
    }

    if (mTtyFd != -1)
        resetTty(mTtyFd, mOldTtyMode);

    delete mBlitter;
}

bool QLinuxFbScreen::initialize()
{
    QRegularExpression ttyRx(QLatin1String("tty=(.*)"));
    QRegularExpression fbRx(QLatin1String("fb=(.*)"));
    QRegularExpression mmSizeRx(QLatin1String("mmsize=(\\d+)x(\\d+)"));
    QRegularExpression sizeRx(QLatin1String("size=(\\d+)x(\\d+)"));
    QRegularExpression offsetRx(QLatin1String("offset=(\\d+)x(\\d+)"));

    QString fbDevice, ttyDevice;
    QSize userMmSize;
    QRect userGeometry;
    bool doSwitchToGraphicsMode = true;

    // Parse arguments
    for (const QString &arg : qAsConst(mArgs)) {
        QRegularExpressionMatch match;
        if (arg == QLatin1String("nographicsmodeswitch"))
            doSwitchToGraphicsMode = false;
        else if (arg.contains(mmSizeRx, &match))
            userMmSize = QSize(match.captured(1).toInt(), match.captured(2).toInt());
        else if (arg.contains(sizeRx, &match))
            userGeometry.setSize(QSize(match.captured(1).toInt(), match.captured(2).toInt()));
        else if (arg.contains(offsetRx, &match))
            userGeometry.setTopLeft(QPoint(match.captured(1).toInt(), match.captured(2).toInt()));
        else if (arg.contains(ttyRx, &match))
            ttyDevice = match.captured(1);
        else if (arg.contains(fbRx, &match))
            fbDevice = match.captured(1);
    }

    if (fbDevice.isEmpty()) {
        fbDevice = QLatin1String("/dev/fb0");
        if (!QFile::exists(fbDevice))
            fbDevice = QLatin1String("/dev/graphics/fb0");
        if (!QFile::exists(fbDevice)) {
            qWarning("Unable to figure out framebuffer device. Specify it manually.");
            return false;
        }
    }

    // Open the device
    mFbFd = openFramebufferDevice(fbDevice);
    if (mFbFd == -1) {
        qErrnoWarning(errno, "Failed to open framebuffer %s", qPrintable(fbDevice));
        return false;
    }

    // Read the fixed and variable screen information
    fb_fix_screeninfo finfo;
    fb_var_screeninfo vinfo;
    memset(&vinfo, 0, sizeof(vinfo));
    memset(&finfo, 0, sizeof(finfo));

    // shikeDebug 
    td_s32 ret;
    ot_vo_layer layer = 0;
    ot_vo_video_layer_attr layer_attr;
    td_bool is_show;
    ot_fb_alpha alpha;
    td_u32 byte_per_pixel = 0;
    ot_fb_colorkey color_key;
    ot_fb_point point = {0, 0};
    static struct fb_bitfield s_a32 = {24, 8, 0};
    static struct fb_bitfield s_r32 = {16, 8, 0};
    static struct fb_bitfield s_g32 = {8,  8, 0};
    static struct fb_bitfield s_b32 = {0,  8, 0};
    static struct fb_bitfield s_r16 = {10, 5, 0};
    static struct fb_bitfield s_g16 = {5, 5, 0};
    static struct fb_bitfield s_b16 = {0, 5, 0};
    static struct fb_bitfield s_a16 = {15, 1, 0};

    memset(&alpha, 0, sizeof(alpha));

    /* all layer support colorkey */
    color_key.enable = TD_TRUE;
    color_key.value = 0x0;
    if (ioctl(mFbFd, FBIOPUT_COLORKEY_GFBG, &color_key) < 0) {
        qErrnoWarning(errno, "FBIOPUT_COLORKEY_GFBG !");
        return false;
    }

    if (ioctl(mFbFd, FBIOPUT_SCREEN_ORIGIN_GFBG, &point) < 0) {
        qErrnoWarning(errno, "set screen original show position failed!");
        return false;
    }

    //alpha.pixel_alpha = TD_FALSE;
    //alpha.global_alpha_en = TD_FALSE;
    alpha.pixel_alpha = TD_TRUE;
    alpha.global_alpha_en = TD_TRUE;
    alpha.global_alpha = 255;
    alpha.alpha0 = 255;
    alpha.alpha1 = 0;
    if (ioctl(mFbFd, FBIOPUT_ALPHA_GFBG, &alpha) < 0) {
        qErrnoWarning(errno, "put alpha info failed!");
        return false;
    }

    if(ioctl(mFbFd, FBIOGET_VSCREENINFO, &vinfo) < 0)
    {
        qErrnoWarning(errno, "Get variable screen info failed!");
    }

    memset(&layer_attr, 0, sizeof(layer_attr));
    ret = ss_mpi_vo_get_video_layer_attr(layer, &layer_attr);
    if (ret != TD_SUCCESS) {
        g_fb_size.width = FHD_WIDTH;
        g_fb_size.height = FHD_HEIGHT;
        qErrnoWarning(errno, "hi_mpi_vo_get_video_layer_attr failed ! set frambuffer to FHD size.");
    } else {
        g_fb_size.width = layer_attr.img_size.width;
        g_fb_size.height = layer_attr.img_size.height;
    }

    if (g_color_format == OT_FB_FORMAT_ARGB1555) {
        vinfo.transp = s_a16;
        vinfo.red    = s_r16;
        vinfo.green  = s_g16;
        vinfo.blue   = s_b16;
        vinfo.bits_per_pixel = 16;
    } else {
        vinfo.transp = s_a32;
        vinfo.red    = s_r32;
        vinfo.green  = s_g32;
        vinfo.blue   = s_b32;
        vinfo.bits_per_pixel = 32;
    }

    vinfo.xres_virtual = g_fb_size.width;
    vinfo.yres_virtual = g_fb_size.height * 2;
    vinfo.xres       = g_fb_size.width;
    vinfo.yres       = g_fb_size.height;
    vinfo.activate     = FB_ACTIVATE_NOW;
    byte_per_pixel    = vinfo.bits_per_pixel/8;

    if (ioctl(mFbFd, FBIOPUT_VSCREENINFO, &vinfo) < 0) {
        qErrnoWarning(errno, "Put variable screen info failed!");
    }

    ot_fb_layer_info layer_info;

    memset(&layer_info, 0, sizeof(layer_info));
    layer_info.buf_mode = OT_FB_LAYER_BUF_DOUBLE;
    layer_info.mask = OT_FB_LAYER_MASK_BUF_MODE;
    if (ioctl(mFbFd, FBIOPUT_LAYER_INFO, &layer_info) < 0) {
        qErrnoWarning(errno, "PUT_LAYER_INFO failed!");
        return false;
    }

    // shikeDebug done

    if (ioctl(mFbFd, FBIOGET_FSCREENINFO, &finfo) != 0) {
        qErrnoWarning(errno, "Error reading fixed information");
        return false;
    }

    if (ioctl(mFbFd, FBIOGET_VSCREENINFO, &vinfo)) {
        qErrnoWarning(errno, "Error reading v   ariable information");
        return false;
    }

    mDepth = determineDepth(vinfo);
    mBytesPerLine = finfo.line_length;
    QRect geometry = determineGeometry(vinfo, userGeometry);
    mGeometry = QRect(QPoint(0, 0), geometry.size());
    mFormat = determineFormat(vinfo, mDepth);
    mPhysicalSize = determinePhysicalSize(vinfo, userMmSize, geometry.size());

    // mmap the framebuffer
    mMmap.size = finfo.smem_len;
    //shikeDebug
    #if 0
    uchar *data = (unsigned char *)mmap(0, mMmap.size, PROT_READ | PROT_WRITE, MAP_SHARED, mFbFd, 0);
    if ((long)data == -1) {
        qErrnoWarning(errno, "Failed to mmap framebuffer");
        return false;
    }
    #endif

    is_show = TD_TRUE;
    if (ioctl(mFbFd, FBIOPUT_SHOW_GFBG, &is_show) < 0) {
        qErrnoWarning(errno, "FBIOPUT_SHOW_GFBG failed!");
        return false;
    }

    if (ss_mpi_sys_mmz_alloc(&g_canvas_addr, &g_buf_addr, TD_NULL, TD_NULL, g_fb_size.width * g_fb_size.height *
        (byte_per_pixel)) == TD_FAILURE) {
        qErrnoWarning(errno, "allocate memory failed!");
        return false;
    }

    g_canvas_buf.canvas.phys_addr = g_canvas_addr;
    g_canvas_buf.canvas.width = g_fb_size.width;
    g_canvas_buf.canvas.height = g_fb_size.height;
    g_canvas_buf.canvas.pitch = g_fb_size.width * byte_per_pixel;
    g_canvas_buf.canvas.format = g_color_format;
    if (memset_s(g_buf_addr, g_fb_size.width * g_fb_size.height * (byte_per_pixel), 0x00, g_canvas_buf.canvas.pitch * g_canvas_buf.canvas.height) != EOK) {
        qErrnoWarning(errno, "memset_s failed!");
        ss_mpi_sys_mmz_free(g_canvas_addr, g_buf_addr);
        g_canvas_addr = 0;
        return false;
    }
    //shikeDebug done

    mMmap.offset = geometry.y() * mBytesPerLine + geometry.x() * mDepth / 8;
    //mMmap.data = data + mMmap.offset;
    mMmap.data = (unsigned char*)g_buf_addr + mMmap.offset;

    QFbScreen::initializeCompositor();
    mFbScreenImage = QImage(mMmap.data, geometry.width(), geometry.height(), mBytesPerLine, mFormat);

    mCursor = new QFbCursor(this);

    mTtyFd = openTtyDevice(ttyDevice);
    if (mTtyFd == -1)
        qErrnoWarning(errno, "Failed to open tty");

    switchToGraphicsMode(mTtyFd, doSwitchToGraphicsMode, &mOldTtyMode);
    blankScreen(mFbFd, false);

    return true;
}

QRegion QLinuxFbScreen::doRedraw()
{
    td_s32 ret;
    QRegion touched = QFbScreen::doRedraw();

    if (touched.isEmpty())
        return touched;

    if (!mBlitter)
        mBlitter = new QPainter(&mFbScreenImage);

    mBlitter->setCompositionMode(QPainter::CompositionMode_Source);
    for (const QRect &rect : touched){
        mBlitter->drawImage(rect, mScreenImage, rect);
    }
        
    g_canvas_buf.update_rect.x = 0;
    g_canvas_buf.update_rect.y = 0;
    g_canvas_buf.update_rect.width = g_canvas_buf.canvas.width;
    g_canvas_buf.update_rect.height = g_canvas_buf.canvas.height;
    ret = ioctl(mFbFd, FBIO_REFRESH, &g_canvas_buf);
    if (ret < 0) {
        qErrnoWarning(errno, "REFRESH failed !");
    }
    
    return touched;
}

// grabWindow() grabs "from the screen" not from the backingstores.
// In linuxfb's case it will also include the mouse cursor.
QPixmap QLinuxFbScreen::grabWindow(WId wid, int x, int y, int width, int height) const
{
    if (!wid) {
        if (width < 0)
            width = mFbScreenImage.width() - x;
        if (height < 0)
            height = mFbScreenImage.height() - y;
        return QPixmap::fromImage(mFbScreenImage).copy(x, y, width, height);
    }

    QFbWindow *window = windowForId(wid);
    if (window) {
        const QRect geom = window->geometry();
        if (width < 0)
            width = geom.width() - x;
        if (height < 0)
            height = geom.height() - y;
        QRect rect(geom.topLeft() + QPoint(x, y), QSize(width, height));
        rect &= window->geometry();
        return QPixmap::fromImage(mFbScreenImage).copy(rect);
    }

    return QPixmap();
}

QT_END_NAMESPACE

#include "moc_qlinuxfbscreen.cpp"

