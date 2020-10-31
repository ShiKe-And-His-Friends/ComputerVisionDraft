#include <stdio.h>

int main () {
	FILE* fp = NULL;
	char buf[255];

	fp = fopen("file_test1.txt" ,"r");
	fscanf(fp ,"%s" ,buf);
	printf("1: %s\n" ,buf);

	fgets(buf ,255 ,(FILE*)fp);
	printf("2: %s\n" ,buf);

	fgets(buf ,255 ,(FILE*)fp);
	printf("3 %s\n" ,buf);

	fgets(buf ,255 ,(FILE*)fp);
	printf("4 %s\n" ,buf);

	fclose(fp);
}
