from src import (
    logger,
    parser,
    Path,
    Manager
)

def main() -> None:
    logger.info("Start application")
    args = parser()

    if args.download:
        Manager.download_dataset(
            author="maysee",
            dataset_name="mushrooms-classification-common-genuss-images", 
            save_loaction=Path.datasets
        )

    
    
    logger.info("Finish application")

if __name__ == "__main__":
    main()