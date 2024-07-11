JL = julia --project

default: init test

init:
	$(JL) -e 'using Pkg; Pkg.precompile(); Pkg.activate("docs"); Pkg.develop(path="."), Pkg.precompile()'

update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.precompile(); Pkg.activate("docs"); Pkg.update(); Pkg.precompile()'

test:
	$(JL) -e 'using Pkg; Pkg.test()'

coverage:
	$(JL) -e 'using Pkg; Pkg.test(; coverage=true)'

serve:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); using LiveServer; servedocs(;skip_dirs=["docs/src/assets", "docs/src/generated"])'

clean:
	rm -rf docs/build
	find . -name "*.cov" -type f -print0 | xargs -0 /bin/rm -f

.PHONY: init test coverage serve clean update