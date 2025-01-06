#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _kaprox_reg();
extern void _kdrca1_reg();
extern void _na3_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," kaprox.mod");
fprintf(stderr," kdrca1.mod");
fprintf(stderr," na3.mod");
fprintf(stderr, "\n");
    }
_kaprox_reg();
_kdrca1_reg();
_na3_reg();
}
