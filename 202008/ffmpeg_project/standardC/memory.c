#include <stdio.h>
#include <assert.h>

void* my_memcpy(void *dest ,const void *str ,int size) {
	assert(dest);
	assert(str);
	assert(size > 0);
	while (size --) {
		*(char *)dest = *(char *)str;
		(char *)dest = (char *)dest + 1;
		(char *)str = (char *)str + 1;
	}
	return dest;
}


int main() {
	char* p = "asdfghjkl";
	char* q = NULL;
	char data[100];

	int len = strlen(p);
	q = (char *)data;

	q = my_memcpy(q ,p ,len);

	for (int i = 0; i < len; i++) {
		printf("%c", *q);
		q++;
	}
	printf("\n");

	return 0;
}
