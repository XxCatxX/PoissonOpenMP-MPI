#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*
 * Un paso del método de Jacobi para la ecuación de Poisson
 *
 *   Argumentos:
 *     - N,M: dimensiones de la malla
 *     - Entrada: x es el vector de la iteración anterior, b es la parte derecha del sistema
 *     - Salida: t es el nuevo vector
 *
 *   Se asume que x,b,t son de dimensión (N+2)*(M+2), se recorren solo los puntos interiores
 *   de la malla, y en los bordes están almacenadas las condiciones de frontera (por defecto 0).
 */

void jacobi_step(int n_local,int M,double *x_local,double *b,double *t, int rank, int numprocs, MPI_Status st)
{
  int i, j, ld=M+2, line_1, line_n, last_line, next, prev;
  line_1 = ld+1;
  line_n = (ld*n_local)+1;
  last_line = (ld*(n_local+1))+1;

  if(rank == 0) prev = MPI_PROC_NULL;
  else prev = rank-1;
  if(rank == numprocs-1) next = MPI_PROC_NULL;
  else next = rank+1; 

  //printf("Process %d in sendrecv\n",rank);

  MPI_Sendrecv(&x_local[line_1], M, MPI_DOUBLE, prev, 0, &x_local[last_line], M, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &st);
  MPI_Sendrecv(&x_local[line_n], M, MPI_DOUBLE, next, 0, &x_local[1], M, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &st);

  for (i=1; i<=n_local; i++) {
    for (j=1; j<=M; j++) {
      t[i*ld+j] = (b[i*ld+j] + x_local[(i+1)*ld+j] + x_local[(i-1)*ld+j] + x_local[i*ld+(j+1)] + x_local[i*ld+(j-1)])/4.0;
      if(rank == 0) printf("%f ",t[i*ld+j]);
    }
    if(rank==0) printf("\n");
  }
  if(rank==0) printf("----------------------------------------------------------\n");
}

/*void jacobi_step_parallel(int n_local,int M,double *x,double *b,double *t)
{
  
  int i, j, ld=M+2;


  for (i=1; i<=n_local; i++) {
    for (j=1; j<=M; j++) {
      t[i*ld+j] = (b[i*ld+j] + x[(i+1)*ld+j] + x[(i-1)*ld+j] + x[i*ld+(j+1)] + x[i*ld+(j-1)])/4.0;
    }
  }
}
*/

/*
 * Método de Jacobi para la ecuación de Poisson
 *
 *   Suponemos definida una malla de (N+1)x(M+1) puntos, donde los puntos
 *   de la frontera tienen definida una condición de contorno.
 *
 *   Esta función resuelve el sistema Ax=b mediante el método iterativo
 *   estacionario de Jacobi. La matriz A no se almacena explícitamente y
 *   se aplica de forma implícita para cada punto de la malla. El vector
 *   x representa la solución de la ecuación de Poisson en cada uno de los
 *   puntos de la malla (incluyendo el contorno). El vector b es la parte
 *   derecha del sistema de ecuaciones, y contiene el término h^2*f.
 *
 *   Suponemos que las condiciones de contorno son igual a 0 en toda la
 *   frontera del dominio.
 */
void jacobi_poisson(int n_local,int M,double *x_local,double *b, double *t, int rank, int numprocs, MPI_Status st)
{
  int i, j, k, ld=M+2, conv, maxit=10000;
  double s, global_s, tol=1e-6;

  k = 0;
  conv = 0;

  while (!conv && k<maxit) {

    /* calcula siguiente vector */
    jacobi_step(n_local, M, x_local, b, t, rank, numprocs, st);

    /* criterio de parada: ||x_{k}-x_{k+1}||<tol */
    s = 0.0;
    for (i=1; i<=n_local; i++) {
      for (j=1; j<=M; j++) {
        s += (x_local[i*ld+j]-t[i*ld+j])*(x_local[i*ld+j]-t[i*ld+j]);
      }
    }
    
    //printf("Process %d in MPI_Reduce\n",rank);
    MPI_Reduce(&s, &global_s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   
    if(rank==0){
      conv = (sqrt(global_s)<tol);
      printf("Error en iteración %d: %g\n", k, sqrt(global_s));
    } 

    //printf("Process %d in MPI_Bcast\n",rank);
    MPI_Bcast(&conv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /* siguiente iteración_local */
    k = k+1;
    for (i=1; i<=n_local; i++) {
      for (j=1; j<=M; j++) {
        x_local[i*ld+j] = t[i*ld+j];
      }
    }
  }
}

int main(int argc, char **argv)
{
  int i, j, N=30, M=30, ld, rank, numprocs;
  double *x, *x_local, *t, *b, h=0.01, f=1.5;


  /* Extracción de argumentos */
  if (argc > 1) { /* El usuario ha indicado el valor de N */
    if ((N = atoi(argv[1])) < 0) N = 50;
  }
  if (argc > 2) { /* El usuario ha indicado el valor de M */
    if ((M = atoi(argv[2])) < 0) M = 1;
  }
  ld = M+2;  /* leading dimension */

  /* Reserva de memoria */
  b = (double*)calloc((N+2)*(M+2),sizeof(double));
  
  /* Inicializar datos */
  for (i=1; i<=N; i++) {
    for (j=1; j<=M; j++) {
      b[i*ld+j] = h*h*f;  /* suponemos que la función f es constante en todo el dominio */
    }
  }
  
  //void jacobi_step_parallel(int n_local,int M,double *x,double *b,double *t)
  
  MPI_Status st;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);  
  
  /*Create n_local y x_local*/
  int n_local = N/numprocs;
  printf("N_LOCAL: %d",n_local);
  //int line_1, line_n, last_line;

  t = (double*)calloc((n_local+2)*(M+2),sizeof(double));
  x_local = (double*)calloc((n_local+2)*(M+2),sizeof(double));

  /* Resolución del sistema por el método de Jacobi */
  jacobi_poisson(n_local,M,x_local,b,t,rank,numprocs, st);

  /*
  printf(" ------------------------------------------------ x_local process %d ------------------------------------------------ \n",rank);
  
  for (i=1; i<=n_local; i++) {
    for (j=1; j<=M; j++) {
      printf("%g ", x_local[i*ld+j]);
    }
    printf("\n");
  }
  printf("------------------------------------------------------------------------------------------------------------------\n");
  printf("Process %d gathering\n",rank); */
  
  if(rank==0) x = (double*)calloc((N+2)*(M+2),sizeof(double));
  
  int gather_size = ld*n_local;
  MPI_Gather(&x_local[ld], gather_size, MPI_DOUBLE, &x[ld], gather_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  free(t);
  free(x_local);
  MPI_Barrier(MPI_COMM_WORLD);

  //jacobi_poisson(N,M,x,b,t);

  /* Imprimir solución (solo para comprobación, eliminar en el caso de problemas grandes) */
  
  if(rank == 0){
    printf("Solution process %d: \n", rank);
    for (i=1; i<=N; i++) {
      for (j=1; j<=M; j++) {
        printf("%g ", x[i*ld+j]);
      }
      printf("\n");
    }
    free(x);
  }

  free(b);
  MPI_Finalize();

  return 0;
}

