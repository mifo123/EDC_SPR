PYTHON ?= python3
PDFLATEX ?= /Library/TeX/texbin/pdflatex
BIBTEX ?= /Library/TeX/texbin/bibtex
PACK_DIR ?= dist
PACK_FILE ?= $(PACK_DIR)/edc_spr_submission.zip
PACK_READY_FILE ?= $(PACK_DIR)/edc_spr_submission_ready.zip

.PHONY: all benchmark shin paper pack pack-ready clean-latex

all: paper

benchmark:
	$(PYTHON) src/generate_benchmark.py

shin: benchmark
	$(PYTHON) src/prepare_shin_real_validation.py
	$(PYTHON) src/zfr_fuzzy_expert.py

paper: shin
	cd paper && $(PDFLATEX) -interaction=nonstopmode main.tex
	cd paper && $(BIBTEX) main
	cd paper && $(PDFLATEX) -interaction=nonstopmode main.tex
	cd paper && $(PDFLATEX) -interaction=nonstopmode main.tex

pack: all
	mkdir -p $(PACK_DIR)
	zip -r -FS $(PACK_FILE) README.md zadanie.md Makefile .gitignore .gitattributes src data outputs paper \
		-x "*.DS_Store" "*/__pycache__/*" "*.pyc" \
		-x "paper/*.aux" "paper/*.bbl" "paper/*.blg" "paper/*.fdb_latexmk" \
		-x "paper/*.fls" "paper/*.log" "paper/*.out" "paper/*.synctex.gz" "paper/*.toc"
	@echo "Created $(PACK_FILE)"

pack-ready:
	mkdir -p $(PACK_DIR)
	zip -r -FS $(PACK_READY_FILE) README.md zadanie.md Makefile .gitignore .gitattributes \
		src/zfr_fuzzy_expert.py \
		data/sample_candidates.csv data/shin_real_candidates.csv data/shin_real_selected_sources.csv \
		outputs paper \
		-x "*.DS_Store" "*/__pycache__/*" "*.pyc" \
		-x "paper/*.aux" "paper/*.bbl" "paper/*.blg" "paper/*.fdb_latexmk" \
		-x "paper/*.fls" "paper/*.log" "paper/*.out" "paper/*.synctex.gz" "paper/*.toc"
	@echo "Created $(PACK_READY_FILE)"

clean-latex:
	find paper -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.blg" -o -name "*.fdb_latexmk" -o -name "*.fls" -o -name "*.log" -o -name "*.out" -o -name "*.synctex.gz" -o -name "*.toc" \) -delete
