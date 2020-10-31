#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int sharedi = 0;
void increase_num(void);

int main () {
	int ret;
	pthread_t thread1 ,thread2 ,thread3;

	ret = pthread_create(&thread1 ,NULL ,(void*)increase_num ,NULL);
	ret = pthread_create(&thread2 ,NULL ,(void*)increase_num ,NULL);
	ret = pthread_create(&thread3 ,NULL ,(void*)increase_num ,NULL);

	pthread_join(thread1 ,NULL);
	pthread_join(thread2 ,NULL);
	pthread_join(thread3 ,NULL);

	printf("sharedi = %d\n" ,sharedi);

	return 0;
}

void increase_num(void) {
	long i,tmp;
	for (i = 0 ; i < 100000 ; i++) {
		tmp = sharedi;
		tmp = tmp + 1;
		sharedi = tmp;
	}
}

