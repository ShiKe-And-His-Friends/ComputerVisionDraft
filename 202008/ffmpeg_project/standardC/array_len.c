#include <stdio.h>

int main() {
	int a[] = {1 ,2 ,3 ,4};
	printf("Int Array:\n");
	printf("array bytes is %d\n" ,sizeof(a));
	printf("first byte size is %d\n" ,sizeof(a + 0));
	printf("first byte address size is %d\n" ,sizeof(*a));
	printf("second byte address size is %d\n" ,sizeof(a + 1));
	printf("second byte size is %d\n" ,sizeof(a[1]));
	printf("array address size is %d\n" ,sizeof(&a));
	printf("array size %d\n", sizeof(*&a));
	printf("next array size is %d\n" ,sizeof(&a + 1));

	char arr[] = {'a' ,'b' ,'c'  ,'d'  ,'e'  ,'f'};

	printf("\nByte Array:\n");
	printf("array bytes is %d\n" ,sizeof(arr));
	printf("first byte size is %d\n", sizeof(arr + 0));
	printf("first byte address size is %d\n", sizeof(*arr));
	printf("second byte size is %d\n", sizeof(arr[1]));
	printf("array address size is %d\n", sizeof(&arr));
	printf("next array size is %d\n", sizeof(&a + 1));
	printf("string length array is %d(random)\n" ,strlen(arr));
	printf("string length array from zero is %d(random)\n", strlen(arr + 0));
	//strlen((char *))
	//printf("%d\n" ,strlen(*arr));
	//printf("%d\n", strlen(arr[1]));
	printf("string length from this array %d (random > 6)\n" ,strlen(&arr));
	printf("string length from array %d (random)\n", strlen(&arr + 1));
	printf("string length from second byte %d (random)\n", strlen(&arr[0] + 1));

	printf("\nByte Array2:\n");

	char lrr[] = "abcdef";
	printf("array bytes is %d\n" ,sizeof(lrr));
	printf("first byte size is %d\n" ,sizeof(lrr + 0));
	printf("first byte address size is %d\n", sizeof(*lrr));
	printf("second byte size is %d\n", sizeof(lrr[1]));
	printf("array address size is %d\n", sizeof(&lrr));
	printf("next array address size is %d\n", sizeof(&lrr + 1));
	printf("second bytes address size is %d\n", sizeof(&lrr[0] + 1));
	printf("string length array is %d\n", strlen(lrr));
	rintf("string length array is %d\n", strlen(lrr + 0));
	// strlen((char *))
	//printf("%d\n" ,strlen(lrr));
	//printf("%d\n", strlen(lrr[1]));
	printf("string length array size is %d\n" ,strlen(&lrr));
	printf("string length after this array %d(random)", strlen(&lrr + 1));
	printf("string length after second byte %d(random)", strlen(&lrr[0] + 1));

	printf("\Char Point:\n");

	char* p = "abcdef";
	printf("array bytes address size is %d\n" ,sizeof(p));
	printf("array second bytes address size is %d\n", sizeof(p + 1));
	printf("first member address size is %d\n", sizeof(*p));
	printf("first member size is %d\n", sizeof(p[0]));
	printf("pointer-to-pointer adddress size is %d\n", sizeof(&p));
	printf("pointer-to-pointer next adddress size is %d\n", sizeof(&p + 1));
	printf("array second bytes address size is %d\n", sizeof(&p[0] + 1));
	printf("string length array is %d\n" ,strlen(p));
	printf("string length array from second member is %d\n", strlen(p + 1));
	printf("string length array from second array is %d(random)\n", strlen(&p + 1));
	printf("string length array from second member is %d\n", strlen(&p[0] + 1));

	printf("\Two Vexter Array:\n");

	int aa[3][4] = {0};

	printf("arrays bytes size is %d\n" ,sizeof(aa));
	printf("first member size is %d\n", sizeof(aa[0][0]));
	printf("first line size is %d\n" ,sizeof(aa[0]));
	printf("second member size is %d\n", sizeof(aa[0] + 1));
	printf("second member in first array size is %d\n", sizeof(aa + 1));
	printf("second member in array size is %d\n", sizeof(&aa[0] + 1));
	printf("arrays size is %d\n", sizeof(*aa));
	printf("last arrays size is %d(EOF)\n", sizeof(aa[3]));
	return 0;
}