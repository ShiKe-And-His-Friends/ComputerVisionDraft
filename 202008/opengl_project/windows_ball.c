#include <GL/glut.h>
#include <stdlib.h>

void init (void) {
	GLfloat mat_specular[] = {1.0 ,1.0 ,1.0 ,1.0};
	GLfloat mat_shininess[] = {50.0};
	GLfloat light_position[] = {1.0 ,1.0 ,1.0 ,0.0};
	glClearColor(0.0 ,0.0 ,0.0 ,0.0);
	glShadeModel(GL_SMOOTH);
	glMaterialfv(GL_FRONT ,GL_SPECULAR ,mat_specular);
	glMaterialfv(GL_FRONT ,GL_SHININESS ,mat_shininess);
	glLightfv(GL_LIGHT0 ,GL_POSITION ,light_position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);
}

void display (void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glutSolidSphere(1.0 ,40 ,50);
	glFlush();
}

int main (int argc ,char ** argv) {
	//GLUT initialize
	glutInit(&argc ,argv);
	//Show pattern initialize
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB |GLUT_DEPTH);
	//Set Windows width and height
	glutInitWindowSize(300 ,300);
	//Set Windows position
	glutInitWindowPosition(100 ,100);
	//Set widnows title bar
	glutCreateWindow(argv[0]);
	//Call OpenGL initializtion funtion
	init();
	//Regist OpenGL draw funciton
	glutDisplayFunc(display);
	/**
	 * Goto GLUT looper.
	 * Start process program
	 * */
	glutMainLoop();
	return 0;
}
