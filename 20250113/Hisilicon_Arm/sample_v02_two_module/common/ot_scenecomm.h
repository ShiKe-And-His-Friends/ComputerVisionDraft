/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */

#ifndef OT_SCENECOMM_H
#define OT_SCENECOMM_H

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include "ot_type.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/* * color log macro define */
#define NONE "\033[m"
#define RED "\033[0;32;31m"
#define GREEN "\033[0;32;32m"
#define BLUE "\033[0;32;34m"
#define YELLOW "\033[1;33m"

/* * product log level define */
#define OT_SCENELOG_LEVEL_FATAL 0
#define OT_SCENELOG_LEVEL_ERROR 1
#define OT_SCENELOG_LEVEL_WARNING 2
#define OT_SCENELOG_LEVEL_DEBUG 3

/* * product log level */
#ifdef CFG_DEBUG_LOG_ON
#define OT_SCENELOG_LEVEL OT_SCENELOG_LEVEL_DEBUG
#else
#define OT_SCENELOG_LEVEL OT_SCENELOG_LEVEL_ERROR
#endif

#ifdef __LITEOS__
#define ot_scene_log(level, module, fmt, args...) do {                                \
        if (level <= OT_SCENELOG_LEVEL) {                                             \
            printf(module "(%s-%d:%d): " fmt, __FUNCTION__, __LINE__, level, ##args); \
        }                                                                             \
    } while (0)
#else
/* * general product log macro define */
#define ot_scene_log(level, module, fmt, args...)  do {                                        \
        if (level <= OT_SCENELOG_LEVEL) {                                                      \
            fprintf(stdout, module "(%s-%d:%d): " fmt, __FUNCTION__, __LINE__, level, ##args); \
        }                                                                                      \
    } while (0)
#endif

/* * product module name, canbe redefined in module */
#ifdef OT_MODULE
#undef OT_MODULE
#endif
#define OT_MODULE "SCENE"

#ifdef CFG_DEBUG_LOG_ON
/* * unified log macro define */
#define scene_loge(fmt, args...) ot_scene_log(OT_SCENELOG_LEVEL_ERROR, OT_MODULE, fmt, ##args)
#define scene_logw(fmt, args...) ot_scene_log(OT_SCENELOG_LEVEL_WARNING, OT_MODULE, fmt, ##args)
#define scene_logd(fmt, args...) ot_scene_log(OT_SCENELOG_LEVEL_DEBUG, OT_MODULE, fmt, ##args)
#else
#define scene_loge(fmt, args...) ot_scene_log(OT_SCENELOG_LEVEL_ERROR, OT_MODULE, fmt, ##args)
#define scene_logw(fmt, args...)
#define scene_logd(fmt, args...)
#endif

/* * \addtogroup     APPCOMM */
/* * @{ */ /* * <!-- [APPCOMM] */

/* * Invalid FD */
#define OT_SCENECOMM_FD_INVALID_VAL (-1)

/* * General String Length */
#define OT_SCENECOMM_COMM_STR_LEN (32 + 1)

/* * Path Maximum Length */
#define OT_SCENECOMM_MAX_PATH_LEN 64

/* * FileName Maximum Length */
#define OT_SCENECOMM_MAX_FILENAME_LEN 64

/* * Pointer Check */
#define ot_scenecomm_check_pointer_return(p, errcode) do { \
        if ((p) == NULL) {                                 \
            scene_loge("null pointer\n");                  \
            return errcode;                                \
        }                                                  \
    } while (0)

/* * Pointer Check return but without return value */
#define ot_scenecomm_check_pointer_return_no_value(p) do { \
        if ((p) == NULL) {                                 \
            scene_loge("null pointer\n");                  \
            return;                                        \
        }                                                  \
    } while (0)

/* * Expression Check */
#define ot_scenecomm_check_expr_return(expr, errcode) do {      \
        if ((expr) == 0) {                                      \
            scene_loge(RED " expr[%s] false" NONE "\n", #expr); \
            return errcode;                                     \
        }                                                       \
    } while (0)

/* * Expression Check */
#define ot_scenecomm_expr_true_return(expr, errcode) do {       \
        if ((expr) == 1) {                                      \
            scene_loge(RED " expr[%s] false" NONE "\n", #expr); \
            return errcode;                                     \
        }                                                       \
    } while (0)

/* * Return Result Check */
#define ot_scenecomm_check_return(ret, errcode) do {     \
        if ((ret) != TD_SUCCESS) {                         \
            scene_loge(RED " ret[%08x]" NONE "\n", ret); \
            return errcode;                              \
        }                                                \
    } while (0)
/* * Return Result UnReturn */
#define ot_scenecomm_check_unreturn(ret) do {            \
        if ((ret) != TD_SUCCESS) {                         \
            scene_loge(RED " ret[%08x]" NONE "\n", ret); \
        }                                                \
    } while (0)


#define ot_scenecomm_check(ret, errcode) do {                                 \
        if ((ret) != TD_SUCCESS) {                                              \
            scene_loge(RED " ret[%08x] errcode[%x]" NONE "\n", ret, errcode); \
        }                                                                     \
    } while (0)


/* * Return Result Check With ErrInformation */
#define ot_scenecomm_check_return_with_errinfo(ret, errcode, errstring) do { \
        if ((ret) != TD_SUCCESS) {                                             \
            scene_loge(RED " [%s] failed[%08x]" NONE "\n", errstring, ret);  \
            return errcode;                                                  \
        }                                                                    \
    } while (0)

/* * Range Check */
#define ot_scenecomm_check_range(value, min, max) (((value) <= (max) && (value) >= (min)) ? 1 : 0)

/* * Memory Safe Free */
#define ot_scenecomm_safe_free(p) do { \
        if ((p) != NULL) {             \
            free(p);                   \
            (p) = NULL;                \
        }                              \
    } while (0)

/* * strcmp enum string and value */
#define ot_scenecomm_strcmp_enum(enumStr, enumValue) strcmp(enumStr, #enumValue)

/* * Mutex Lock */
#ifndef ot_mutex_init_lock
#define ot_mutex_init_lock(mutex) do {          \
        (void)pthread_mutex_init(&(mutex), NULL); \
    } while (0)
#endif
#ifndef ot_mutex_lock
#define ot_mutex_lock(mutex) do {         \
        (void)pthread_mutex_lock(&(mutex)); \
    } while (0)
#endif
#ifndef ot_mutex_unlock
#define ot_mutex_unlock(mutex) do {         \
        (void)pthread_mutex_unlock(&(mutex)); \
    } while (0)
#endif
#ifndef ot_mutex_destroy
#define ot_mutex_destroy(mutex) do {         \
        (void)pthread_mutex_destroy(&(mutex)); \
    } while (0)
#endif

/* * SCENE Error BaseId : [28bit~31bit] unique */
#define OT_SCENECOMM_ERR_BASEID (0x80000000L)

/* * SCENE Error Code Rule [ --base[4bit]--|--module[8bit]--|--error[20bit]--]
 * module : module enum value [OT_APP_MOD_E]
 * event_code : event code in specified module, unique in module
 */
#define OT_SCENECOMM_ERR_ID(module, err) ((td_s32)((OT_SCENECOMM_ERR_BASEID) | ((module) << 20) | (err)))

/* * App Module ID */
typedef enum {
    OT_SCENE_MOD_SCENE = 0,
    OT_SCENE_MOD_CONFACCESS,
    OT_SCENE_MOD_BUTT
} ot_scene_mod;

#ifndef scene_array_size
#define scene_array_size(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* End of #ifndef OT_APPCOMM_H */
