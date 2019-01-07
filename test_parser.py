import argparse
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        help="train batch size", type=bool,default=False)
    args = parser.parse_args()
    print(args.batch_size)


if __name__ == '__main__':
    train()