#include <stdio.h>
#include <Windows.h>

int average(int n , ...) {
	va_list arg;
	int i = 0;
	int sum = 0;
	va_start(arg ,n);
	for (; i < n ; i++) {
		sum += va_arg(arg ,int);
	}
	return sum / n;
	va_end(arg);
}

int main() {
	int a = 1;
	int b = 2;
	int c = 3;
	int avg1 = average(2 ,a ,b);
	int avg2 = average(3, a, b ,c);
	printf("%d\n" ,avg1);
	printf("%d\n", avg2);
	system("pause");
	return 0;
}