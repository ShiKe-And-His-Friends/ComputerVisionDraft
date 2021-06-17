#include <stdio.h>
#include <stdlib.h>

/*
	4
	0.499154
	1.996614
	16
	0.249577
	3.993229
	999999
	0.000998
	998.304016
	1111111
	0.000948
	1053.417236
*/

static float invSqrt(float x) {
	float xhalf = 0.5 * x;
	int i = *(int *)&x;
	i = 0x5f3759df - (i >> 1);
	x = *(float *) & i;
	x = x * (1.5 - xhalf * x * x);
	return x;
}

int main() {

	char buf[1024];
	while (fgets(buf ,sizeof(buf) ,stdin) != NULL) {
		int length = strlen(buf) - 1;
		if (length >= 10) {
			printf("input values of int[2^31 - 1 ,-2^31]\n");
			continue;
		}
		int index = length - 1;
		float multiple = 10;
		float num = buf[index] - 48;
		for (index -- ; index >= 0 ; index --)
		{
			num += (buf[index] - 48) * multiple;
			multiple *= 10;
		}
		
		float inv_sqrt = invSqrt(num);
		printf("%f\n" ,inv_sqrt);

		float sqrt = invSqrt(num) * num;
		printf("%f\n", sqrt);
	}

	return 0;
}