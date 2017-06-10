all: calcGAP addPred add3Pred labPerf predCorr ensemble gaddPred gadd3Pred

calcGAP: calcGAP.cpp pred.cpp pred.hpp utils.cpp utils.hpp
	g++ -std=gnu++11 -pthread calcGAP.cpp pred.cpp utils.cpp -o calcGAP

addPred: addPred.cpp pred.cpp pred.hpp utils.cpp utils.hpp
	g++ -std=gnu++11 -pthread addPred.cpp pred.cpp utils.cpp -o addPred

add3Pred: add3Pred.cpp pred.cpp pred.hpp utils.cpp utils.hpp
	g++ -std=gnu++11 -pthread add3Pred.cpp pred.cpp utils.cpp -o add3Pred

gaddPred: gaddPred.cpp pred.cpp pred.hpp utils.cpp utils.hpp
	g++ -std=gnu++11 -pthread gaddPred.cpp pred.cpp utils.cpp -o gaddPred

gadd3Pred: gadd3Pred.cpp pred.cpp pred.hpp utils.cpp utils.hpp
	g++ -std=gnu++11 -pthread gadd3Pred.cpp pred.cpp utils.cpp -o gadd3Pred

labPerf: labelPerformance.cpp pred.cpp pred.hpp utils.cpp utils.hpp
	g++ -std=gnu++11 -pthread labelPerformance.cpp pred.cpp utils.cpp -o labPerf 

predCorr: predCorr.cpp pred.cpp corr.cpp utils.cpp utils.hpp
	g++ -std=gnu++11 -pthread predCorr.cpp pred.cpp corr.cpp utils.cpp -o predCorr

ensemble: ensemble.cpp pred.cpp pred.hpp utils.cpp utils.hpp
	g++ -g -std=gnu++11 -pthread ensemble.cpp pred.cpp utils.cpp -o ensemble

files2Cov: files2cov.cpp pred.cpp corr.cpp utils.cpp
	g++ -std=gnu++11 -pthread files2cov.cpp corr.cpp pred.cpp utils.cpp -o files2Cov

# local scripts below
scribble: scribble.cpp pred.cpp pred.hpp utils.cpp utils.hpp
	g++ -std=gnu++11 -pthread scribble.cpp pred.cpp utils.cpp -o scribble 

