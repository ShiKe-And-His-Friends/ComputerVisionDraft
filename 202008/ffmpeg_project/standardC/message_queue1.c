#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#define KEY 1024

struct msg_buf {
	long type; // values must > 0
	char data[128];
};

//typedef struct msg_buf buf;

int main () {
	printf("ENTER \"end\" to exit program...\n");

	int msg_id ,ret;
	pid_t pid;
	struct msg_buf buf;
	msg_id = msgget(KEY ,IPC_CREAT | IPC_EXCL);
	if (msg_id == -1) {
		perror("msg get failed");
		exit(1);
	}

	pid = fork();

	if (pid == -1 ) {
		perror("fork error");
		exit(1);
	} else if (pid == 0) {
		while (1) {
			scanf("%s" ,buf.data);
			buf.type = 1;
			ret = msgsnd(msg_id ,&buf ,sizeof(buf.data) ,0);
			if (ret == -1) {
				perror("message send");
				exit(1);
			}
			if (strncmp(buf.data ,"end" ,3) == 0) {
				buf.type = 2;
				msgsnd(msg_id ,&buf ,sizeof(buf.data) ,0);
				break;
			}
			memset(buf.data ,0 ,sizeof(buf.data));
		}
	} else {
		while (1) {
			memset(buf.data ,0 ,sizeof(buf.data));
			ret = msgrcv(msg_id ,&buf ,sizeof(buf.data) ,2 ,0);
			if (ret == -1) {
				perror("message receive");
				exit(1);
			}
			if (strncmp(buf.data ,"end" ,3) == 0) {
				kill(pid ,SIGKILL);
				break;
			}
			printf("receive msg:\t\t %s \n" ,buf.data);
		}
		waitpid(pid ,NULL ,0);
	}
	msgctl(msg_id ,IPC_RMID ,NULL);

	return 0;
}
