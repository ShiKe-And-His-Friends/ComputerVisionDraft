#include <stdio.h>
#include <Windows.h>

int check_sys() {

	union {
		int i;
		char c;
	}un;

	un.i = 1;
	return un.c;
}

int main() {
	int ret = check_sys();
	if (ret == 1) {
		printf("Ð¡¶Ë\n");
	}
	else {
		printf("´ó¶Ë\n");
	}
	system("pause");
	return 0;
}