#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>

#define MAXSIZE 10

int stack[MAXSIZE];
int size = 0;
sem_t sem;

//producer
void provide_data(void) {
	int i;
	for (i = 0 ; i < MAXSIZE ;i++) {
		stack[i] = i;
		sem_post(&sem);
	}
}

//consumer
void handle_data(void) {
	int i;
	while ((i = size ++) < MAXSIZE) {
		sem_wait(&sem);
		printf("\nThe multiplication %d X %d = %d \n" ,stack[i] ,stack[i] ,stack[i]*stack[i]);
		sleep(1);
	}
}

int main (void) {
	pthread_t provider ,handler;

	sem_init(&sem ,0 ,0);
	pthread_create(&provider ,NULL ,(void*)provide_data ,NULL);
	pthread_create(&handler ,NULL ,(void*)handle_data ,NULL);

	pthread_join(handler ,NULL);
	pthread_join(provider ,NULL);

	sem_destroy(&sem);

	return 0;
}

