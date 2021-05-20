#include <stdio.h>
#include <assert.h>

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
		printf("YYYYYYYYY\n", size);
		while (size --) {
			printf("%d\n", size);
			*(char *)dest = *(char *)str;
			printf("%c\n", *(char*)str);
			printf("%c\n", *(char*)dest);
			(char *)dest = (char *)dest + 1;
			(char *)str = (char *)str + 1;
		}
	} else {
		printf("NNNNNNNNN\n", size);
		dest = (char *)dest + size - 3;
		str = (char*)str + size - 3;
		while (size--) {
			printf("%d\n" ,size);
			printf("%c\n", *(char*)str);
			*(char*)dest = *(char*)str;
			printf("%c\n", *(char*)dest);
			(char*)dest = (char*)dest - 1;
			(char*)str = (char*)str - 1;
		}
	}
	return result;
}

int main() {
	char* p = NULL;
	char* q = NULL;
	char data[100];
	char data2[100] = "asdfghjkl";
	
	int len = 9;
	//int len = strlen(p);
	q = (char *)data;
	p = (char*)data2;

	//p = (q + 1);
	q = (p - 1);
	printf("%p\n", p);
	printf("%p\n", q);
	/*char tt = q;
	for (int i = 0; i < len; i++) {
		printf("%c", *q);
		q++;
	}
	printf("\n");
	q = tt;*/
	printf("%p\n", p);
	printf("%p\n", q);
	q = my_memmove(q ,p ,len);
	printf("\n");
	printf("NNN");
	printf("\n");
	q = my_memmove(p ,q ,len);
	printf("XXX");
	printf("\n");

	return 0;
}
