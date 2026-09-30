# Reproduce the figures of Wadekar et al. (PRL).  Run from the repository root.
#
#   make figures             all seven figures from the shipped data/  (~30 s)
#   make test                the test suite                            (~2 min)
#   make simulate-fast       regenerate the Fig. 3 and caustics data   (~10 min)
#   make simulate            regenerate every data product    (~60 CPU-hours)
#   make compare             compare regenerated/ against data/
#   make figures DATA=regenerated   plot from regenerated data instead
#
# Variables: PYTHON (interpreter), NPROC (worker processes), OUT (where
# simulations write), DATA (what the figures read).

PYTHON ?= python
NPROC  ?= 16
OUT    ?= regenerated
DATA   ?= data

SIM  = $(PYTHON) scripts/simulate
PLOT = LENSING_DATA=$(DATA) $(PYTHON) scripts/plot

.PHONY: figures fig1 fig2 fig3 fig4 fig5 fig6 fig7 test \
        simulate simulate-fast compare clean

figures: fig1 fig2 fig3 fig4 fig5 fig6 fig7

fig1: ; $(PLOT)/plot_prl_effectualness.py
fig2: ; $(PLOT)/plot_prl_chisq.py
fig3: ; $(PLOT)/plot_prl_strategies.py
fig4: ; $(PLOT)/plot_prl_appendix_lenses.py --which pointmass
fig5: ; $(PLOT)/plot_prl_appendix_lenses.py --which cr
fig6: ; $(PLOT)/plot_prl_appendix_lenses.py --which sis
fig7: ; $(PLOT)/plot_prl_appendix_lenses.py --which cored

test:
	$(PYTHON) -m pytest -q

# --- data products --------------------------------------------------------

simulate-fast:
	$(SIM)/lensed_strategies.py --scan intrinsic --a-min 2.0 \
	    --out $(OUT)/strategies/lensed_strategies_rhoUL.npz
	LENSING_STRATEGIES=$(OUT)/strategies/lensed_strategies_rhoUL.npz \
	    $(SIM)/appendix_caustics.py --out $(OUT)/caustics/appendix_caustics.json

simulate: simulate-fast
	@echo "Regenerating every data product: about 60 CPU-hours in total."
	# Fig. 1(a, c)                                      ~0.5 h on 20 cores
	$(SIM)/phase3_wave_optics.py --m-tot 50 --seg-dur 16 --grid-n-td 2318 \
	    --grid-n-y 8 --t-d-max 0 --n-proc $(NPROC) --outdir $(OUT)/pointmass_M50
	# Fig. 1(b, d), needs GLoW                          ~0.3 h on 16 cores
	$(SIM)/appendix_chang_refsdal.py --m-lz 3000 --n-proc $(NPROC) \
	    --outdir $(OUT)/chang_refsdal
	# Fig. 2                                            ~0.4 h on 16 cores
	$(SIM)/appendix_chisq_consistency.py --m-tot 50 --n-proc $(NPROC) \
	    --outdir $(OUT)/chisq
	$(SIM)/appendix_chisq_background.py --n-seg 300000 --seed 20260903 \
	    --n-proc $(NPROC) --outdir $(OUT)/chisq
	# Fig. 4                                            ~0.8 h on 16 cores
	$(SIM)/phase3_wave_optics.py --m-tot 11 --seg-dur 32 --grid-n-td 1000 \
	    --grid-n-y 8 --n-proc $(NPROC) --outdir $(OUT)/pointmass_M11_M100
	$(SIM)/phase3_wave_optics.py --m-tot 100 --seg-dur 16 --grid-n-td 1000 \
	    --grid-n-y 8 --n-proc $(NPROC) --outdir $(OUT)/pointmass_M11_M100
	# Fig. 5, needs GLoW                                ~0.5 h on 16 cores
	$(SIM)/appendix_chang_refsdal.py --m-lz 3000 --extend-a --n-proc $(NPROC) \
	    --outdir $(OUT)/chang_refsdal_extended_a
	# Figs. 6 and 7 and Table II, need GLoW             ~1.4 h on 16 cores
	$(SIM)/appendix_lens_models.py --lens-model sis --m-tot 11 --seg-dur 32 \
	    --n-proc $(NPROC) --outdir $(OUT)/isothermal
	$(SIM)/appendix_lens_models.py --lens-model sis --m-tot 50 --seg-dur 16 \
	    --n-proc $(NPROC) --outdir $(OUT)/isothermal
	$(SIM)/appendix_lens_models.py --lens-model sis --m-tot 100 --seg-dur 16 \
	    --n-proc $(NPROC) --outdir $(OUT)/isothermal
	for rc in 0.05 0.15 0.3; do \
	  $(SIM)/appendix_lens_models.py --lens-model cored --core-radius $$rc \
	      --m-tot 50 --seg-dur 16 --n-proc $(NPROC) --outdir $(OUT)/isothermal; \
	done

compare:
	$(PYTHON) scripts/checks/compare_data.py data $(OUT)

clean:
	rm -rf $(OUT)
