#include "llist.h"

linklist create_null_list_link(void) {
	linklist llist = (linklist)malloc(sizeof(struct node));
	if (llist != NULL) {
		llist->next = NULL;
	}
	else {
		fprintf(stderr ,"out of space !\n");
	}
	return llist;
}