#include <stdio.h>

int main () {

	//w+ pattern
	FILE* filePtr1 = NULL;
	filePtr1 = fopen("file_test1.txt" ,"w+");

	// fprintf
	fprintf(filePtr1 ,"This is "
			"Demo in file.h w+ __PATTERN__\n"
			"HAHA");
	fprintf(filePtr1 ,"\nfprintf end\n");

	// fputs	
	fputs("This is "
		"Demo in file.h w+ __PARTTERN__\n"
		"HAHA" ,filePtr1);
	fputs("\nfputs\n" ,filePtr1);
	
	fclose(filePtr1);

}
