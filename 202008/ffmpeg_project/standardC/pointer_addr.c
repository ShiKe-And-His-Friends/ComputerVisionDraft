#include <stdio.h>
#include <assert.h>

int pointer_foot_long() {
	int n = 10;
	char* pc = (char*)&n; //pointer to int lower byte address
	int* pi = &n;
	printf("%p\n" ,&n);
	printf("%p\n", pc);
	printf("%p\n", pc + 1);
	printf("%p\n", pi);
	printf("%p\n", pi + 1);

	return 0;
}

int pointer_soft_value() {
	int n = 0x11223344;
	char* pc = (char*)&n;
	int *pi = &n;
	*pc = 0x55;
	printf("%p\n" ,pc);
	printf("%x\n", n);
	*pi = 0;
	printf("%p\n", pi);
	printf("%x\n", n);

	return 0;
}

int pointer_to_pointer_add_value() {
	int a = 10;
	int *pa = &a;
	int **ppa = &pa;
	printf("%p\n" ,&pa);
	printf("%x\n", ppa);

	**ppa = 30;
	printf("%x\n", a);
	printf("%x\n", ppa);

	return 0;
}

#define N 5
int clear_arrray() {
	float f[N];
	float* p;
	for (p = &f[0]; p < &f[N];) {
		*p++ = 0;
	}
}

int clear_byte_array() {
	float f[N];
	float* p;
	for (p = &f[N - 1]; p >= &f[0]; p--) {
		*p = 0;
	}
	return 0;
}

void reverse(int *a ,int size) {
	assert(a);
	int* start = a;
	int* end = a + size - 1;
	while (start < end) {
		// a != b
		*start ^= *end;
		*end ^= *start;
		*start ^= *end;
		start++;
		end--;
	}
}

int main() {
	int a[10] = {1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,0};
	int size = sizeof(a) / sizeof(a[0]);
	reverse(a ,size);
	for (int i = 0; i < size; i++) {
		printf("%d ",a[i]);
	}
	return 0;
}