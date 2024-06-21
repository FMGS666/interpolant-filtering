## displaying current directory
echo Running the experiment on the gaussian ssm with sine non linearity;

#### NO SCHEDULER SIM PREPROC PFFP_v0
## without burn-in
python -m int_filt --pp-before-interpolant --full-out --log-results;
python -m int_filt --pp-before-interpolant --b-net-amortized --full-out --log-results;
python -m int_filt --full-out --log-results;
python -m int_filt --b-net-amortized --full-out --log-results;

## with burn-in
python -m int_filt --pp-before-interpolant --full-out --log-results --num-burn-in-steps 100000;
python -m int_filt --pp-before-interpolant --b-net-amortized --full-out --log-results --num-burn-in-steps 100000;
python -m int_filt --full-out --log-results --num-burn-in-steps 100000;
python -m int_filt --b-net-amortized --full-out --log-results --num-burn-in-steps 100000;

#### NO SCHEDULER SIM PREPROC PFFP_v1
## without burn-in
python -m int_filt --pp-before-interpolant --full-out --log-results --interpolant-method pffp_v1;
python -m int_filt --pp-before-interpolant --b-net-amortized --full-out --log-results --interpolant-method pffp_v1;
python -m int_filt --full-out --log-results --interpolant-method pffp_v1;
python -m int_filt --b-net-amortized --full-out --log-results --interpolant-method pffp_v1;

## with burn-in
python -m int_filt --pp-before-interpolant --full-out --log-results --num-burn-in-steps 100000 --interpolant-method pffp_v1;
python -m int_filt --pp-before-interpolant --b-net-amortized --full-out --log-results --num-burn-in-steps 100000 --interpolant-method pffp_v1;
python -m int_filt --full-out --log-results --num-burn-in-steps 100000 --interpolant-method pffp_v1;
python -m int_filt --b-net-amortized --full-out --log-results --num-burn-in-steps 100000 --interpolant-method pffp_v1;

#### COSINE ANNEALING SCHEDULER SIM PREPROC PFFP_v0
## without burn-in
python -m int_filt --pp-before-interpolant --full-out --log-results --b-net-scheduler cosine-annealing;
python -m int_filt --pp-before-interpolant --b-net-amortized --full-out --log-results --b-net-scheduler cosine-annealing;
python -m int_filt --full-out --log-results --b-net-scheduler cosine-annealing;
python -m int_filt --b-net-amortized --full-out --log-results --b-net-scheduler cosine-annealing;

## with burn-in
python -m int_filt --pp-before-interpolant --full-out --log-results --num-burn-in-steps 100000 --b-net-scheduler cosine-annealing;
python -m int_filt --pp-before-interpolant --b-net-amortized --full-out --log-results --num-burn-in-steps 100000 --b-net-scheduler cosine-annealing;
python -m int_filt --full-out --log-results --num-burn-in-steps 100000 --b-net-scheduler cosine-annealing;
python -m int_filt --b-net-amortized --full-out --log-results --num-burn-in-steps 100000 --b-net-scheduler cosine-annealing;

#### COSINE ANNEALING SIM PREPROC PFFP_v1
## without burn-in
python -m int_filt --pp-before-interpolant --full-out --log-results --interpolant-method pffp_v1 --b-net-scheduler cosine-annealing;
python -m int_filt --pp-before-interpolant --b-net-amortized --full-out --log-results --interpolant-method pffp_v1 --b-net-scheduler cosine-annealing;
python -m int_filt --full-out --log-results --interpolant-method pffp_v1 --b-net-scheduler cosine-annealing;
python -m int_filt --b-net-amortized --full-out --log-results --interpolant-method pffp_v1 --b-net-scheduler cosine-annealing;

## with burn-in
python -m int_filt --pp-before-interpolant --full-out --log-results --num-burn-in-steps 100000 --interpolant-method pffp_v1 --b-net-scheduler cosine-annealing;
python -m int_filt --pp-before-interpolant --b-net-amortized --full-out --log-results --num-burn-in-steps 100000 --interpolant-method pffp_v1 --b-net-scheduler cosine-annealing;
python -m int_filt --full-out --log-results --num-burn-in-steps 100000 --interpolant-method pffp_v1 --b-net-scheduler cosine-annealing;
python -m int_filt --b-net-amortized --full-out --log-results --num-burn-in-steps 100000 --interpolant-method pffp_v1 --b-net-scheduler cosine-annealing;



