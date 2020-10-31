#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void print_message_fuction (void* ptr);

int main () {

	int tmp1 ,tmp2;
	void *retval;
	pthread_t thread1 ,thread2;
	char* message1 = "pthread1";
	char* message2 = "pthread2";

	int ret_thread1 ,ret_thread2;
	ret_thread1 = pthread_create(&thread1 ,NULL ,(void*)&print_message_fuction ,(void*) message1);
	ret_thread2 = pthread_create(&thread2 ,NULL ,(void*)&print_message_fuction ,(void*)message2);

	if (!ret_thread1) {
		printf("thread 1 created succes.\n");
	} else {
		printf("thread 1 created failed.\n");
	}
	if (!ret_thread2) {
		printf("thread 2 created success.\n");
	} else {
		printf("thread 2 created failed.\n");
	}
	printf("\n\n");

	tmp1 = pthread_join(thread1 ,&retval);
	printf("thread 1 return value(retval) is %d\n" ,(int)retval);
	printf("thread 1 return result is %d\n" ,tmp1);
	if (tmp1) {
		printf("thread 1 can not join.\n");
	}
	printf("thread 1 end.\n");

	tmp2 = pthread_join(thread1 ,&retval);
	printf("thread 2 return value(retval) is %d\n" ,(int)retval);
	printf("thread 2 return result is %d\n" ,tmp2);
	if (tmp2) {
		printf("thread 2 can not join.\n");
	}
	printf("thread 2 end.\n");

}

void print_message_fuction(void *ptr) {
	int i = 0;
	for (i = 0 ; i < 5 ; i++) {
		printf("%s:%d\n" ,(char*)ptr ,i);
	}
}
