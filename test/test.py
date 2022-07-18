import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--tmp", help = "tmp") 
    
    args = parser.parse_args()
    
    print(f"args.tmp : {args.tmp}")
    
    
    