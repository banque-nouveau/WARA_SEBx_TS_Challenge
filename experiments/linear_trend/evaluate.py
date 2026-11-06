import argparse
import json

from amgm.data import load_test_data
from amgm.utils import calculate_metrics
from amgm.models import mymodel

def predict_submission(inputs):
    model = mymodel.model()
    return model.predict(inputs)

def evaluate_submission(predictions, targets):
    return calculate_metrics(targets, predictions)

def main():
    parser = argparse.ArgumentParser(description="Evaluate the residual regression submission")
    parser.add_argument("--data_source", default="Data/data-20250901", help="Directory containing security_data.csv or security_data.txt")
    parser.add_argument("--start-date", default="2018-12-31")
    parser.add_argument("--end-date", default="2020-12-31")
    parser.add_argument("--num-iids", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading data from data source: {args.data_source}")
    test_data = load_test_data(
        args.data_source, start_date=args.start_date, end_date=args.end_date,
        num_iids=args.num_iids, seed=args.seed,
    )
    
    print("Running predictions on test data...")
    predictions = predict_submission(test_data.inputs)
    
    print("Evaluating predictions...")
    print(json.dumps(evaluate_submission(predictions, test_data.targets), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
