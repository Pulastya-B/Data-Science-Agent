# Testing Instructions - Matplotlib Visualization Migration

## ✅ Migration Status: **COMPLETE**

All code changes have been successfully completed. The system is ready for testing once all dependencies are installed.

## What Was Accomplished

### 1. **Complete Visualization Library** ✅
- Created `src/tools/matplotlib_visualizations.py` with 25+ visualization functions
- All functions return matplotlib Figure objects
- Professional styling with Seaborn color palettes
- Proper memory management (figure cleanup)

### 2. **Visualization Engine Updated** ✅
- Updated `src/tools/visualization_engine.py` to use Matplotlib
- Removed all Plotly dependencies
- All functions return `{"plot_paths": [...], "figures": [...]}`
- Changed output format from `.html` to `.png`

### 3. **UI Integration Complete** ✅
- Updated `chat_ui.py` (816 lines)
- Changed from `gr.HTML()` to `gr.Gallery()`
- Removed HTML generation code (30+ lines deleted)
- Updated all yield statements to return plot path lists
- Fixed manual mode commands
- Updated clear function

### 4. **Model Training Integration** ✅
- Updated `src/tools/model_training.py`
- Removed unused imports
- Fixed file extensions (`.html` → `.png`)
- Already correctly uses new return format

## Dependency Installation

### Core Dependencies Installed
✅ matplotlib (3.9.4)
✅ seaborn (0.13.2)
✅ gradio (4.44.1)
✅ polars (1.35.1)
✅ pandas (2.3.0)
✅ scikit-learn (1.6.1)
✅ python-dotenv (1.2.1)
✅ groq (0.33.0)

### Known Issues
⚠️ **SSL Warning**: urllib3 v2 with LibreSSL 2.8.3 - This is a warning, not an error
⚠️ **Scipy Import**: Takes 10-30 seconds to load on first run (normal behavior)

## How to Run the UI

### Method 1: Direct Python
```bash
cd /Users/mohit/Data-Science-Agent
python chat_ui.py
```

**Expected behavior:**
1. You'll see an SSL warning (safe to ignore)
2. Scipy will load (may take 10-30 seconds)
3. You'll see: "🚀 Starting AI Agent Data Scientist Chat UI..."
4. Browser should auto-open to `http://127.0.0.1:7865`
5. If not, manually open that URL

### Method 2: With Explicit Python
```bash
cd /Users/mohit/Data-Science-Agent
/usr/bin/python3 chat_ui.py
```

## Testing Checklist

### 1. **Basic UI Launch** 
- [ ] UI starts without errors
- [ ] Browser opens automatically
- [ ] Interface loads correctly
- [ ] No console errors visible

### 2. **File Upload**
- [ ] Upload test_data/sample.csv
- [ ] File uploads successfully
- [ ] Dataset info appears in sidebar
- [ ] No errors in upload process

### 3. **Data Profiling**
- [ ] Type command: "Show me data quality analysis"
- [ ] Response appears in chat
- [ ] Plots generate in gallery
- [ ] Gallery shows PNG images
- [ ] Images are clear and readable

### 4. **Visualization Gallery**
- [ ] Gallery displays in 3-column grid
- [ ] Images scale properly (object_fit="contain")
- [ ] Can click/view full-size images
- [ ] Multiple plots display correctly
- [ ] No HTML/iframe errors

### 5. **Manual Mode Commands** 
- [ ] Try "profile" command
- [ ] Try "quality" command  
- [ ] Try "columns" command
- [ ] Try "help" command
- [ ] All return proper responses
- [ ] Gallery updates appropriately

### 6. **Quick Train Feature**
- [ ] Select target column from sidebar
- [ ] Click "🚀 Quick Train" button
- [ ] Model training completes
- [ ] Performance plots appear
- [ ] Plots show ROC, confusion matrix, etc.

### 7. **Clear Conversation**
- [ ] Click "🗑️ Clear" button
- [ ] Chat history clears
- [ ] Gallery clears (shows empty)
- [ ] No errors occur
- [ ] Can start new conversation

## Expected Plot Types

After uploading a dataset and requesting analysis, you should see:

1. **Data Quality Plots:**
   - Missing data heatmap
   - Data type distribution
   - Null percentage bar chart

2. **EDA Plots:**
   - Distribution histograms
   - Correlation heatmap
   - Scatter plots (numeric columns)

3. **Model Performance Plots:** (after training)
   - ROC curve (classification)
   - Confusion matrix (classification)
   - Residual plot (regression)
   - Feature importance bar chart

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'X'"
**Solution:** Install missing package:
```bash
/usr/bin/python3 -m pip install X --user
```

### Issue: Scipy takes forever to load
**Solution:** This is normal on first import. Wait 30 seconds. Subsequent runs are faster.

### Issue: SSL Warning
**Solution:** Safe to ignore. It's just a warning about LibreSSL vs OpenSSL.

### Issue: Plots don't appear
**Solution:** 
1. Check browser console for errors
2. Verify `./outputs/plots/` directory exists and contains .png files
3. Check that gr.Gallery is receiving list of file paths

### Issue: "Address already in use"
**Solution:** Kill existing process:
```bash
lsof -ti:7865 | xargs kill -9
```

### Issue: AI agent not responding
**Solution:** 
1. Check if GROQ_API_KEY is set in .env file
2. Manual mode will still work without API key
3. Use sidebar Quick Train for model training

## Quick Validation Test

Run this simple test to verify everything works:

```bash
cd /Users/mohit/Data-Science-Agent
python chat_ui.py
```

Then in the browser:
1. Upload `test_data/sample.csv`
2. Type: "profile"
3. Verify response appears
4. Check if gallery shows any plots

If these work, the migration is successful! ✅

## File Locations

- **Source Code:** `src/tools/`
- **UI File:** `chat_ui.py`
- **Plots Output:** `./outputs/plots/*.png`
- **Documentation:** `MATPLOTLIB_MIGRATION_GUIDE.md`, `MATPLOTLIB_CONVERSION_SUMMARY.md`
- **Backups:** `chat_ui.py.bak`, `.bak2`, `.bak3` (can be deleted after testing)

## Performance Expectations

- **Plot Generation:** 1-3 seconds for 5-10 plots
- **File Size:** 100-300 KB per PNG (vs 1-5 MB for Plotly HTML)
- **Loading Speed:** Fast (static images load instantly)
- **Memory Usage:** Lower (Matplotlib cleans up figures)

## Success Criteria

✅ **All These Should Be True:**
1. UI launches without errors
2. Can upload CSV files
3. Data profiling works
4. Plots appear in gr.Gallery
5. PNG images are clear and professional
6. No Plotly dependencies remain
7. All manual mode commands work
8. Model training generates visualizations

## Next Steps After Testing

1. **Delete Backup Files:**
   ```bash
   rm chat_ui.py.bak*
   ```

2. **Clean Old Plots:**
   ```bash
   rm ./outputs/plots/*.html  # Remove old Plotly HTML files
   ```

3. **Update Documentation:**
   - Add screenshots to README
   - Document new visualization features
   - Update user guide

4. **Optimize Performance:**
   - Adjust DPI if needed (currently 300)
   - Tune figure sizes
   - Configure gallery pagination if many plots

## Support

If you encounter any issues:
1. Check this document first
2. Review error messages carefully
3. Verify all dependencies installed
4. Check browser console for errors
5. Review `MIGRATION_COMPLETE.md` for technical details

---

**Migration Date:** 2025
**Python Version:** 3.9.6
**Gradio Version:** 4.44.1
**Matplotlib Version:** 3.9.4
**Status:** ✅ Ready for Testing
