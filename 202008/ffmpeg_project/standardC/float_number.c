#include <stdio.h>

int main() {
	int n = 9;
	float* p = (float*)&n;
	printf("n int : %d\n" ,n);
	printf("*p : %f\n" ,*p);
	*p = 9;
	printf("n int : %d\n", n);
	printf("*p : %f\n", *p);
	return 0;
}