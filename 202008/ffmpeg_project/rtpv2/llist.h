#ifndef _L_LIST_H
#define _L_LIST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAX_SEND_FAIL_N
#define MAX_SEND_FAIL_N 29
#endif

typedef struct _client_info {
	char ipaddr[16];
	int socket_c;
} client_info;

struct node {
	client_info node_info;
	int sned_fail_n;
	struct node* next;
};

typedef struct node* pnode;
typedef struct node* linklist;

linklist create_null_list_link(void);

int is_nulllist_link(linklist llist);

linklist insert_link(linklist llist ,const char* ipaddr);

linklist delete_node(linklist llist ,const char* ipaddr);

pnode delete_this_node(linklist llist ,pnode this_pnode);

pnode search_node(linklist llist ,const char* ipaddr);

int num_node(linklist llist);

int insert_nodulp_node(linklist llist ,const char* ipaddr);

int free_linklist(linklist llist);

#endif