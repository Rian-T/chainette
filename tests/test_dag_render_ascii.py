from chainette.utils.dag import build_rich_tree, RenderOptions
from chainette.core.branch import Branch
from chainette import Chain, apply


def _dummy_fn(x):
    return [x]

inc = apply(_dummy_fn)  # type: ignore[arg-type]

br = Branch(name="br", steps=[inc])
chain = Chain(name="test", steps=[[br, br]])

def test_ascii_tree(capsys):
    opts = RenderOptions(icons_on=False, max_branches=1)
    from rich.console import Console
    console = Console(force_terminal=True, width=60)
    console.print(build_rich_tree(chain, opts=opts))
    out = capsys.readouterr().out
    assert '+1 more' in out or 'more…' in out
    # ensure no emoji present
    assert '📄' not in out 