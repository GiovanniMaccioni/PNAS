#
To run PNAS:
    pnas_main.py --pnas
#
To run RANDOM SEARCH:
    pnas_main.py --num_blocks b

Note: each run will save in the specified folder, a couple of tensors cells, accuracies. To perform the next experiment you have to save in the same folder different runs varying
        the num_blocks parameter starting from 1, with step 1, to the desired number of blocks (example b = 1, b = 2, b = 3)

#
After the results obtained from RANDOM SEARCH, EVALUATE CORRELATION can be run:

    evaluate_correlation_main.py --num_blocks b

Note: b has to coincide with the number of experiments ran in the previous experiment as well as the folder to which save the correlation plots