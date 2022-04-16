#include "stdio.h"
#include "malloc.h"

#define MAX 500

main() {
	int* p, n = 0, max = MAX;
	FILE* fp;
	clrscr();
	fp = fopen("d:\\aa.txt","r");
	p = (int*)malloc(MAX * sizeof(int));

	while (!feof(fp)) {
		if (n == max) {
			max = max + MAX;
			brk(p + max);
		}
		fscanf(fp ,"%d" ,&p[n]);
		printf("%d" ,p[n]);
		n++;
	}
}