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
int is_nulllist_link(linklist llist) {
	return llist->next == NULL;
}

pnode search_node(linklist llist, const char* ipaddr) {
	pnode q = llist->next;
	if (is_nulllist_link(llist)) {
		return NULL;
	}
	do {
		if (!strcmp(q->node_info.ipaddr ,ipaddr)) {
			return q;
		}
		q = q->next;
	} while (q != NULL);
	return NULL;
}

linklist insert_link(linklist llist, const char* ipaddr) {
	pnode q = (pnode)malloc(sizeof(struct node));
	if (q == NULL) {
		fprintf(stderr ,"out of space.\n");
	}
	else {
		strcpy(q->node_info.ipaddr ,ipaddr);
		q->next = llist->next;
		llist->next = q;
	}
	return llist;
}

int insert_nodulp_node(linklist llist, const char* ipaddr) {
	int ret = 0;
	if (search_node(llist ,ipaddr) == NULL) {
		insert_link(llist ,ipaddr);
		ret = 1;
	}
	return ret;
}