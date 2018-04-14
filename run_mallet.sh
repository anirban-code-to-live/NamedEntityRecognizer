echo "Running mallet for file : $1"

java -cp "C:\Users\Rony\Documents\IISC-Study\Sem-II\NLU\Assignments\Ass-III\mallet-2.0.8/class;C:\Users\Rony\Documents\IISC-Study\Sem-II\NLU\Assignments\Ass-III\mallet-2.0.8/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger --train true --test lab --threads 2 --iterations 10 $1