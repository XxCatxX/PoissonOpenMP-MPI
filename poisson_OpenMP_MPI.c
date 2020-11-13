#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "omp.h"

/*
 * Un paso del método de Jacobi para la ecuación de Poisson
 *
 *   Argumentos:
 *     - N,M: dimensiones de la malla
 *     - Entrada: x es el vector de la iteragit inición anterior, b es la parte derecha del sistema
 *     - Salida: t es el nuevo vector
 *
 *   Se asume que x,b,t son de dimensión (N+2)*(M+2), se recorren solo los puntos interiores
 *   de la malla, y en los bordes están almacenadas las condiciones de frontera (por defecto 0).
 */

void jacobi_step(int N,int M,double *x,double *b,double *t)
{
  int i, j, ld=M+2;
  for (i=1; i<=N; i++) {
    for (j=1; j<=M; j++) {
      t[i*ld+j] = (b[i*ld+j] + x[(i+1)*ld+j] + x[(i-1)*ld+j] + x[i*ld+(j+1)] + x[i*ld+(j-1)])/4.0;
    }
  }
}

/* Starting the parallel implementation */
void jacobi_step_parallel(int N,int M,double *x,double *b,double *t)
{
  //obtain rank to define neighbours and size to define last neighbour
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n_local = N/size;
  MPI_Status status;

  int i, j, ld=M+2;

  int below, above;
  /*
  if(rank!=0){
    above=rank-1;}
  else{above=MPI_PROC_NULL;}
  if(rank!=size-1){below=rank+1;}else{below=MPI_PROC_NULL;}
*/

  if(rank == 0) above = MPI_PROC_NULL;
  else above = rank-1;
  if(rank == size-1) below = MPI_PROC_NULL;
  else below = rank+1; 

  //part done with the help of Joaquin
  int x_pos=n_local*rank*ld;
  int send_above, send_below, recv_above, recv_below;
  send_above=x_pos+ld;
  send_below=x_pos+n_local*ld;
  recv_above=x_pos;
  recv_below=x_pos+(n_local+1)*ld;

  MPI_Sendrecv(&x[send_above], ld, MPI_DOUBLE, above, 0, &x[recv_below], ld, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &status);
  MPI_Sendrecv(&x[send_below], ld, MPI_DOUBLE, below, 0, &x[recv_above], ld, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &status);
  
  omp_set_num_threads(4);
  #pragma omp parallel for private(j) schedule(static)
  for (i=1+n_local*rank; i<1+n_local*(rank+1); i++) {
    for (j=1; j<=M; j++) {
      //compute t(i,j) as in the sequential algorithm
      t[i*ld+j] = (b[i*ld+j] + x[(i+1)*ld+j] + x[(i-1)*ld+j] + x[i*ld+(j+1)] + x[i*ld+(j-1)])/4.0;
    }
  }
}

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
void jacobi_poisson(int N,int M,double *x,double *b)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int i, j, k, ld=M+2, conv, maxit=10000;
  double *t, s, tol=1e-6;
  int n_local=N/size;

  t = (double*)calloc((N+2)*(M+2),sizeof(double));

  k = 0;
  conv = 0;

  while (!conv && k<maxit) {

    /* calcula siguiente vector */
    jacobi_step_parallel(N,M,x,b,t);

    /* criterio de parada: ||x_{k}-x_{k+1}||<tol */
    s = 0.0;
    for (i=n_local*rank+1; i<=n_local*(rank+1); i++) {
      for (j=1; j<=M; j++) {
        s += (x[i*ld+j]-t[i*ld+j])*(x[i*ld+j]-t[i*ld+j]);
      }
    }

    MPI_Allreduce(&s, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    conv = (sqrt(s)<tol);

    /* siguiente iteración */
    k = k+1;
    for (i=n_local*rank+1; i<=n_local*(rank+1); i++) {
      for (j=1; j<=M; j++) {
        x[i*ld+j] = t[i*ld+j];
      }
    }

  }

  free(t);

  MPI_Gather(&x[(n_local*rank+1)*ld], n_local*ld, MPI_DOUBLE, &x[ld], n_local*ld, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int i, j, N = 50, M = 50, ld;
  double *x, *b, h = 0.01, f = 1.5, t1, t2;

  /* Extracción de argumentos */
  if (argc > 1){ /* El usuario ha indicado el valor de N */
    if ((N = atoi(argv[1])) < 0) N = 50;
  }
  if (argc > 2){ /* El usuario ha indicado el valor de M */
    if ((M = atoi(argv[2])) < 0) M = 1;
  }
  ld = M + 2; /* leading dimension */

  /* Reserva de memoria */
  x = (double *)calloc((N + 2) * (M + 2), sizeof(double));
  b = (double *)calloc((N + 2) * (M + 2), sizeof(double));

  /* Inicializar datos */
  for (i=1; i<=N; i++){
    for (j=1; j<=M; j++){
      b[i*ld+j] = h*h*f; /* suponemos que la función f es constante en todo el dominio */
    }
  }

  /* Resolución del sistema por el método de Jacobi */
  t1=MPI_Wtime();
  jacobi_poisson(N, M, x, b);
  t2=MPI_Wtime();
  if(rank==0) printf("Time: %f\n",t2-t1);

  /* Imprimir solución (solo para comprobación, eliminar en el caso de problemas grandes) 
  if (!rank)
    for (i=1; i<=N; i++){
      for (j = 1; j <= M; j++){
        printf("%15g ", x[i*ld+j]);
      }
      printf("\n");
    }
*/

  free(x);
  free(b);

  MPI_Finalize();
  return 0;
}
