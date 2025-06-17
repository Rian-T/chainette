import textwrap
from chainette.yaml_loader import load_chain
from chainette import Step, Branch, SamplingParams, Chain
from pydantic import BaseModel


class A(BaseModel):
    q: str

class B(BaseModel):
    a: str

qa = Step(id="qa", name="qa", input_model=A, output_model=B, engine_name="e", sampling=SamplingParams())
branch = Branch(name="br", steps=[qa])
br = branch  # alias for YAML reference


def test_yaml_loader(tmp_path):
    yml = textwrap.dedent(
        """
        name: YAML Test
        batch_size: 1
        steps:
          - qa
          - - br
        """
    )
    file = tmp_path / "chain.yml"
    file.write_text(yml)
    ch: Chain = load_chain(file, symbols=globals())
    assert ch.name == "YAML Test"
    assert len(ch.steps) == 2


def test_yaml_invalid(tmp_path):
    bad = """
    steps:
      - - 123   # not string
    """
    f = tmp_path / "bad.yml"
    f.write_text(bad)
    import pytest
    from jsonschema import ValidationError

    with pytest.raises(ValidationError):
        load_chain(f, symbols=globals()) 