from torchtitan.components.validate import BaseValidator


def test_should_validate_on_first_and_frequency_steps():
    validator = BaseValidator(BaseValidator.Config(freq=10))

    assert validator.should_validate(1)
    assert validator.should_validate(10)
    assert not validator.should_validate(9)


def test_should_validate_on_last_step_even_when_not_frequency_step():
    validator = BaseValidator(BaseValidator.Config(freq=10))

    assert validator.should_validate(9, last_step=True)
