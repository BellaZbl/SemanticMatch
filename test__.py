import argparse
import ast
parser = argparse.ArgumentParser()
parser.add_argument("--new_vocab",
                    help="vocab path", type=ast.literal_eval, default=None)
args = parser.parse_args()

print(args.new_vocab,type(args.new_vocab))
