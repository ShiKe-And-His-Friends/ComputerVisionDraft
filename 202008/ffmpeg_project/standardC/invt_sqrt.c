#include <stdio.h>
#include <stdlib.h>

int main() {

	char buf[1024];
	while (fgets(buf ,sizeof(buf) ,stdin) != NULL) {
		int length = strlen(buf) - 1;
		if (length >= 10) {
			printf("input values of int[2^31 - 1 ,-2^31]\n");
			continue;
		}
		int index = length - 1;
		int multiple = 10;
		int num = buf[index] - 48;
		for (index -- ; index >= 0 ; index --)
		{
			num += (buf[index] - 48) * multiple;
			multiple *= 10;
		}
	
	}

	return 0;
}