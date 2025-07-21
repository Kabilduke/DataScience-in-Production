from loguru import logger

from build.Model_test import UseModel


@logger.catch
def main() -> None:
    logger.info('Starting the model prediction process...')
    DL = UseModel()
    DL.load()
    pred = DL.predict([25, 24, 5, 0.90, 3, 1.5, 50, 6])
    logger.info(f'Prediction: {pred}')


if __name__ == '__main__':
    main()

