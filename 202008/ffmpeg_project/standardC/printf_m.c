#include <stdio.h>
#include <windows.h>
#include <assert.h>

is_print(int n) {
	while (n>0) {
		is_print(n / 10);
		putchar(n % 10 + '0');
	}
}

int printf_m(const char *format , ...) {
	assert(format);
	va_list arg;		
	va_start(arg, format);
	char *p = format;
	while (*p){
		if (*p == '%') {
			p++;
		}
		else {
			putchar(*p);
		}
		p++;
	}
	switch (*p){
	    case 'd':
			is_print(va_arg(arg, int));
			break;

	    case 's':{
			char *start = va_arg(arg, char*);  
			while (*start){
				putchar(*start);
				start++;
			}
		}; 
			break;

		case 'c':
			putchar(va_arg(arg, char));
			break;

		default:
			break;
	}
	return format;
}

int main() {
	printf_m("w\n");
	printf_m("1234\n");
	printf_m("abcdef\n");

	system("pause");
	return 0;
}