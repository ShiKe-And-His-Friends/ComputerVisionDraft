/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */

#include "sample_ipc.h"
#include <pthread.h>
#include <sys/msg.h>
#include "sample_comm.h"

#define SAMPLE_MSG_KEY  1234
#define SAMPLE_MSG_MODE 0660

static int g_server_msg_id = -1;
static int g_client_msg_id = -1;

static pthread_t g_server_msg_thd;
static td_bool g_is_thread_run = TD_FALSE;

static sample_ipc_msg_proc g_msg_proc_func = TD_NULL;

static void *server_msg_handler(void *param)
{
    sample_ipc_msg_req_buf msg_req_buf;
    sample_ipc_msg_res_buf msg_res_buf;
    td_bool is_need_fb = TD_FALSE;
    const long msg_type = -(SAMPLE_MSG_TYPE_RES_NO - 1);

    while (g_is_thread_run == TD_TRUE) {
        /* receive req_data, request type message */
        (td_void)memset_s(&msg_req_buf, sizeof(msg_req_buf), 0, sizeof(msg_req_buf));
        if (msgrcv(g_server_msg_id, &msg_req_buf, sizeof(msg_req_buf.msg_data), msg_type, MSG_NOERROR) == -1) {
            sample_print("receive request data failed!\n");
            return NULL;
        }
        if (g_msg_proc_func == TD_NULL) {
            continue;
        }
        (td_void)memset_s(&msg_res_buf, sizeof(msg_res_buf), 0, sizeof(msg_res_buf));
        if (g_msg_proc_func(&msg_req_buf, &is_need_fb, &msg_res_buf) != TD_SUCCESS) {
            sample_print("msg proc failed!\n");
        }
        if (is_need_fb == TD_TRUE) {
            /* send res_data */
            if (msgsnd(g_server_msg_id, &msg_res_buf, sizeof(msg_res_buf.msg_data), IPC_NOWAIT) == -1) {
                sample_print("send response data failed!\n");
                return NULL;
            }
        }
    }

    return NULL;
}

/* sample IPC server, use system V message queue */
td_s32 sample_ipc_server_init(sample_ipc_msg_proc msg_proc_func)
{
    if (msg_proc_func == TD_NULL) {
        sample_print("msg_proc_func is NULL!\n");
        return TD_FAILURE;
    }

    g_server_msg_id = msgget((key_t)SAMPLE_MSG_KEY, IPC_CREAT | IPC_EXCL | SAMPLE_MSG_MODE);
    if (g_server_msg_id == -1) {
        sample_print("msgget failed!\n");
        return TD_FAILURE;
    }

    g_msg_proc_func = msg_proc_func;
    g_is_thread_run = TD_TRUE;
    if (pthread_create(&g_server_msg_thd, NULL, server_msg_handler, NULL) != 0) {
        g_is_thread_run = TD_FALSE;
        g_msg_proc_func = TD_NULL;
        msgctl(g_server_msg_id, IPC_RMID, NULL);
        g_server_msg_id = -1;
        sample_print("pthread_create failed!\n");
        return TD_FAILURE;
    }

    return TD_SUCCESS;
}

td_void sample_ipc_server_deinit(td_void)
{
    if (g_is_thread_run == TD_TRUE) {
        g_is_thread_run = TD_FALSE;
        pthread_cancel(g_server_msg_thd);
        pthread_join(g_server_msg_thd, NULL);
    }
    g_msg_proc_func = TD_NULL;

    if (g_server_msg_id != -1) {
        if (msgctl(g_server_msg_id, IPC_RMID, NULL) == -1) {
            sample_print("msgctl with IPC_RMID failed!\n");
        }
        g_server_msg_id = -1;
    }
}

/* sample IPC client */
td_s32 sample_ipc_client_init(td_void)
{
    g_client_msg_id = msgget((key_t)SAMPLE_MSG_KEY, SAMPLE_MSG_MODE);
    if (g_client_msg_id == -1) {
        sample_print("msgget failed!\n");
        return TD_FAILURE;
    }

    return TD_SUCCESS;
}

td_void sample_ipc_client_deinit(td_void)
{
    g_client_msg_id = -1;
}

td_s32 sample_ipc_client_send_data(const sample_ipc_msg_req_buf *msg_req_buf,
    td_bool is_need_fb, sample_ipc_msg_res_buf *msg_res_buf)
{
    if (msg_req_buf == TD_NULL) {
        sample_print("msg_req_buf is NULL!\n");
        return TD_FAILURE;
    }
    /* send req_data */
    if (msgsnd(g_client_msg_id, msg_req_buf, sizeof(msg_req_buf->msg_data), IPC_NOWAIT) == -1) {
        sample_print("send request data failed!\n");
        return TD_FAILURE;
    }

    if (is_need_fb == TD_TRUE) {
        if (msg_res_buf == TD_NULL) {
            sample_print("msg_res_buf is NULL!\n");
            return TD_FAILURE;
        }
        /* receive res_data, specified type(msg_type) message */
        if (msgrcv(g_client_msg_id, msg_res_buf, sizeof(msg_res_buf->msg_data),
            msg_res_buf->msg_type, MSG_NOERROR) == -1) {
            sample_print("receive response data failed!\n");
            return TD_FAILURE;
        }
    }

    return TD_SUCCESS;
}
