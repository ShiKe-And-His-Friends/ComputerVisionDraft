static void pgm_save (unsigned char *buf ,int wrap 
		,int xsize ,int ysize ,char *filename) {

	FILE *f;
	int i;
	f = fopen(filename ,"w");
	fprintf(f ,"P5\n%d %d\n %d\n" ,xsize ,ysize ,255);
	for(i = 0 ; i < ysize ; i++) {
		fwrite(buf + i *wrap ,1 ,xsize ,f);
	}
	fclose(f);
}
