#include <stdio.h>
#include <malloc.h>
#include <assert.h>

#define N_SPACE_NUMBER 1024

void* my_memcpy(void *dest ,void *str ,int size) {
	// dest = str + 1 ; crash
	assert(dest);
	assert(str);
	assert(size > 0);
	void* result = NULL;
	result = dest;

	while (size --) {
		*(char *)dest = *(char *)str;
		(char *)dest = (char *)dest + 1;
		(char *)str = (char *)str + 1;
	}
	// except '\0'

	return result;
}

void* my_memmove(void *dest ,void *str ,int size) {
	assert(dest);
	assert(str);
	assert(size > 0);

	void* result = dest;

	if (dest <= str || (char*)dest >= (char*)str + size) {
		printf("From header coverage.\n");
		while (size --) {
			*(char *)dest = *(char *)str;
			(char *)dest = (char *)dest + 1;
			(char *)str = (char *)str + 1;
		}
	} else {
		printf("From tailer coverage.\n");
		dest = (char *)dest + size - 1;
		str = (char*)str + size - 1;
		while (size--) {
			*(char*)dest = *(char*)str;
			(char*)dest = (char*)dest - 1;
			(char*)str = (char*)str - 1;
		}
	}
	return result;
}

void* mymemchr(const void *dest ,char c, int size) {
	assert(dest);
	assert(size > 0);
	while (size && *(char *)dest) {
		if (*(char *)dest == (char) c) {
			return (char *)dest;
		}
		((char*)dest)++;
		size--;
	}
	return 0;
}

int my_mem() {
	char* p = NULL;
	char* q = NULL;
	char data[10] = { "asdfghjkl" };
	char data2[10];
	
	// initilize
	p = (char*)data;
	//q = (char *)data2;

	int len = strlen(p);

	q = (p + 1);
	printf("%p\n", p);
	printf("%p\n", q);

	// solve bytes coverage
	q = my_memmove(q ,p ,len);
	char* tt = q;
	for (int i = 0; i < len; i++) {
		printf("%c", *q);
		q++;
	}
	q = tt;
	printf("\n");

	q = mymemchr(q, 'k' ,len);
	printf("letter is %c\n" ,*q);
	return 0;
}

int continuous_space() {
	int* ptr = NULL;
	ptr = (int*)malloc(N_SPACE_NUMBER * sizeof(int));
	assert(ptr);
	for (int i = 0; i < N_SPACE_NUMBER; i++) {
		*(ptr + i) = 0;
	}
	// do something
	*(ptr + 1) = -1;
	*(ptr + 2) = 1;
	printf("%p\n" ,ptr);
	printf("%d\n" ,ptr[0]);
	printf("%d\n" ,*(ptr + 1));
	printf("%X\n", *(ptr + 1));
	printf("%d\n" ,*(&ptr[0] + 2));
	
	free(ptr);
	ptr = NULL;
	return 0;
}

void test_realloc() {
	int* p = malloc(100);
	assert(p);
	if (p) {

	}
	// OOM
	p = realloc(p , N_SPACE_NUMBER);
	
	int ptr = NULL;
	ptr = realloc(p , N_SPACE_NUMBER);
	assert(ptr);
	if (ptr) {
		p = ptr;
	}
	free(p);
	p = NULL;
}

int main() {
	test_realloc();
	return 0;
}
