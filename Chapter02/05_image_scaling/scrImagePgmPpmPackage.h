#ifndef SCRIMAGEPGMPPMPACKAGE_H
#define SCRIMAGEPGMPPMPACKAGE_H
#include <stdio.h>

 int scr_read_pgm( char* name, unsigned char* image, int irows, int icols );
 void scr_write_pgm( char* name, unsigned char* image, int rows, int cols, char* comment );
 int scr_read_ppm( char* name, unsigned char* image, int irows, int icols );
 void scr_write_ppm( char* name, unsigned char* image, int rows, int cols, char* comment );
 // void get_PgmPpmParams(char * , int *, int *);
 void get_PgmPpmParamsx(char* name, int *irows, int *icols ); 
 void getout_comment(FILE * );


#endif SCRIMAGEPGMPPMPACKAGE_H