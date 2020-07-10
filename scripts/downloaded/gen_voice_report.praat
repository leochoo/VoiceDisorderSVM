name$ = selected$ ("Sound")


form voice report
	optionmenu Pitch_analysis_mode: 3
		option ac method, costs = standard
		option ac method, costs = 0
		option cc method, costs = standard
		option cc method, costs = 0
		option Leave settings as they are
	optionmenu F0_range: 1
		option MDVP 70-625
		option MDVP 200-1000
		option User specified
	real left_User_specified_F0_range 75
	real right_User_specified_F0_range 500
	optionmenu Analysis_time_range: 1
		option Total duration
		option Total duration minus first and last 100 ms
		option User specified
	real left_User_specified_time_range 0
	real right_User_specified_time_range 3
	real Maximum_period_factor 2
	real Maximum_amplitude_factor 9
	optionmenu Saving: 1
		option Don't save
		option Manually from info window's File menu
		option Automatically to Filename = Sound name
		option One line append to user specified file, no header
		option One line append to user specified file, with header
	sentence Save_directory /Users/leochoo/dev/PRAAT
	word User_specified_file_name info.txt
	boolean Draw_waveform_and_pitch_contour 1
endform

if draw_waveform_and_pitch_contour = 1
	Erase all
endif

# set pitch range -----------------------------------------------------------------
if f0_range = 1
	minimum_pitch = 70
	maximum_pitch = 625
elsif f0_range = 2
	minimum_pitch = 200
	maximum_pitch = 1000
elsif f0_range = 3
	minimum_pitch = left_User_specified_F0_range
	maximum_pitch = right_User_specified_F0_range
endif
#echo 'minimum_pitch' 'maximum_pitch'


# set time range ------------------------------------------------------------------


start_of_signal = Get starting time
end_of_signal = Get finishing time

if analysis_time_range = 1
	start = start_of_signal
	end = end_of_signal
elsif analysis_time_range = 2
	start = start_of_signal + 0.1
	end = end_of_signal - 0.1
	if start>= end
		exit Analysis time range too short
	endif
elsif analysis_time_range = 3
	if right_User_specified_time_range > end_of_signal or left_User_specified_time_range < start_of_signal
		exit analysis start precedes signal start or analysis end exceeds signal end
	endif
	start = left_User_specified_time_range
	end = right_User_specified_time_range
endif
duration_total = end_of_signal - start_of_signal
duration_analysed = end - start

#pitch analysis settings -------------------------------------------------------------
pitch_silence_threshold = 0.03
pitch_voicing_threshold = 0.45
if pitch_analysis_mode = 1
	pitch_octave_cost = 0.01
	pitch_octave_jump_cost = 0.35
	pitch_voiced_unvoiced_cost = 0.14
	To Pitch (ac)... 0 minimum_pitch 15 no pitch_silence_threshold pitch_voicing_threshold 
	...0.01 0.35 0.14 maximum_pitch
elsif pitch_analysis_mode = 2
	pitch_octave_cost = 0
	pitch_octave_jump_cost = 0
	pitch_voiced_unvoiced_cost = 0
	To Pitch (ac)... 0 minimum_pitch 15 no pitch_silence_threshold pitch_voicing_threshold 
	...0 0 0 maximum_pitch
elsif pitch_analysis_mode = 3
	pitch_octave_cost = 0.01
	pitch_octave_jump_cost = 0.35
	pitch_voiced_unvoiced_cost = 0.14
	To Pitch (cc)... 0 minimum_pitch 15 no pitch_silence_threshold pitch_voicing_threshold 
	...0.01 0.35 0.14 maximum_pitch
elsif pitch_analysis_mode = 4
	pitch_octave_cost = 0
	pitch_octave_jump_cost = 0
	pitch_voiced_unvoiced_cost = 0
	To Pitch (cc)... 0 minimum_pitch 15 no pitch_silence_threshold pitch_voicing_threshold 
	...0 0 0 maximum_pitch
endif
select Sound 'name$'
plus Pitch 'name$'
To PointProcess (cc)
select Sound 'name$'
plus Pitch 'name$'
plus PointProcess 'name$'_'name$'
#Voice report... start end minimum_pitch maximum_pitch maximum_period_factor maximum_amplitude_factor 0.03 0.45

report$ = Voice report... start end minimum_pitch maximum_pitch maximum_period_factor maximum_amplitude_factor 0.03 0.45

medianPitch = extractNumber (report$, "Median pitch: ")
meanPitch = extractNumber (report$, "Mean pitch: ")
sdPitch = extractNumber (report$, "Standard deviation: ")
minPitch = extractNumber (report$, "Minimum pitch: ")
maxPitch = extractNumber (report$, "Maximum pitch: ")
nPulses = extractNumber (report$, "Number of pulses: ")
nPeriods = extractNumber (report$, "Number of periods: ")
meanPeriod = extractNumber (report$, "Mean period: ") * 1000
sdPeriod = extractNumber (report$, "Standard deviation of period: ") * 1000
pctUnvoiced = extractNumber (report$, "Fraction of locally unvoiced frames: ")*100
fracUnvoiced$ = extractLine$ (report$, "Fraction ")
nVoicebreaks = extractNumber (report$, "Number of voice breaks: ")
pctVoicebreaks = extractNumber (report$, "Degree of voice breaks: ") * 100
degreeVoicebreaks$ = extractLine$ (report$, "Degree ")
jitter_loc = extractNumber (report$, "Jitter (local): ") * 100
jitter_loc_abs = extractNumber (report$, "Jitter (local, absolute): ") * 1000000
jitter_rap = extractNumber (report$, "Jitter (rap): ") * 100
jitter_ppq5 = extractNumber (report$, "Jitter (ppq5): ") *100
shimmer_loc = extractNumber (report$, "Shimmer (local): ") *100
shimmer_loc_dB = extractNumber (report$, "Shimmer (local, dB): ")
shimmer_apq3 = extractNumber (report$, "Shimmer (apq3): ") * 100
shimmer_apq5 = extractNumber (report$, "Shimmer (apq5): ") * 100
shimmer_apq11 = extractNumber (report$, "Shimmer (apq11): ") * 100
mean_autocor = extractNumber (report$, "Mean autocorrelation: ")
mean_nhr = extractNumber (report$, "Mean noise-to-harmonics ratio: ")
mean_hnr = extractNumber (report$, "Mean harmonics-to-noise ratio: ")

date$ = date$()
echo                                VOICE REPORT          
printline -------------------------------------------------------------------------------------------------------------------------
printline Name of analysed sound: 'name$'        date: 'date$'
printline -------------------------------------------------------------------------------------------------------------------------
printline Analysis settings
if pitch_analysis_mode = 1 or pitch_analysis_mode = 3
	printline     Voice analysis (cc method) from 'minimum_pitch' to 'maximum_pitch' Hz
	method$ = "cc"
else
	printline     Intonation analysis (ac method) from 'minimum_pitch' to 'maximum_pitch' Hz
	method$ = "ac"
endif

printline     Oct cost = 'pitch_octave_cost:2', oct jmp cost = 'pitch_octave_jump_cost:2', voi/unvoi cost = 'pitch_voiced_unvoiced_cost:2'
printline     Max period factor = 'maximum_period_factor:2', max amp factor = 'maximum_amplitude_factor:2'
printline     Total duration: from 'start_of_signal:3' to 'end_of_signal:3' = 'duration_total:3' secs
printline     Duration analysed: from 'start:3' to 'end:3' = 'duration_analysed:3' secs

printline Fundamental frequency
printline     Median F0: 'medianPitch:3' Hz
printline     Mean F0: 'meanPitch:3' Hz
printline     St.dev. F0: 'sdPitch:3' Hz
printline     Minimum F0: 'minPitch:3' Hz
printline     Maximum F0: 'maxPitch:3' Hz
printline Pulses
printline     Number of pulses: 'nPulses'
printline     Number of periods: 'nPeriods'
printline     Mean period: 'meanPeriod:3' millisec.
printline     St.dev. period: 'sdPeriod:3' millisec.
printline Voicing
printline     Fraction 'fracUnvoiced$'
printline     Number of voice breaks: 'nVoicebreaks'
printline     Degree 'degreeVoicebreaks$'
printline Jitter
printline     Jitter (local): 'jitter_loc:3' %
printline     Jitter (local, abs): 'jitter_loc_abs:3' microseconds
printline     Jitter (rap): 'jitter_rap:3' %
printline     Jitter (ppq5): 'jitter_ppq5:3' %
printline Shimmer
printline     Shimmer (local): 'shimmer_loc:3' %
printline     Shimmer (local, dB): 'shimmer_loc_dB:3' dB
printline     Shimmer (apq3): 'shimmer_apq3:3' %
printline     Shimmer (apq5): 'shimmer_apq5:3' %
printline     Shimmer (apq11): 'shimmer_apq11:3' %
printline Harmonicity
printline     Mean autocorrelation: 'mean_autocor:4'
printline     Mean NHR: 'mean_nhr:4'
printline     Mean HNR: 'mean_hnr:3' dB
printline -------------------------------------------------------------------------------------------------------------------------


# Draw waveform and pitch contour for control
if draw_waveform_and_pitch_contour = 1
	Select outer viewport... 0 6 0 3
	select Sound 'name$'
	Draw... start end -1 1 yes curve
	select Pitch 'name$'
	lo = Get minimum... 0 0 Hertz Parabolic
	lo = lo - 20
	hi = Get maximum... 0 0 Hertz Parabolic
	hi = hi + 20
	Select outer viewport... 0 6 3 5.5
	Draw... start end lo hi yes
	Select outer viewport... 0 6 0 5.5
endif

if saving > 2 and draw_waveform_and_pitch_contour = 1
	pause Press continue if you want to save
endif
if saving = 1
	printline info not saved
elsif saving = 3
	fappendinfo 'save_directory$'/'name$'.txt
	printline Info appended to 'save_directory$'/'name$'.txt
elsif saving = 4
	fileappend "'save_directory$'\'user_specified_file_name$'" 'name$''tab$''method$''tab$'
	...'minimum_pitch''tab$''maximum_pitch''tab$''pitch_silence_threshold''tab$'
	...'pitch_voicing_threshold''tab$''pitch_octave_cost''tab$''pitch_octave_jump_cost''tab$'
	...'pitch_voiced_unvoiced_cost''tab$'
	...'maximum_period_factor''tab$''maximum_amplitude_factor''tab$'
	...'duration_total''tab$''duration_analysed'
	...'tab$''medianPitch''tab$''meanPitch''tab$''sdPitch''tab$''minPitch''tab$'
	...'maxPitch''tab$''nPulses''tab$''nPeriods''tab$''meanPeriod''tab$''sdPeriod''tab$'
	...'pctUnvoiced''tab$''nVoicebreaks''tab$''pctVoicebreaks''tab$''jitter_loc''tab$'
	...'jitter_loc_abs''tab$''jitter_rap''tab$''jitter_ppq5''tab$''shimmer_loc''tab$'
	...'shimmer_loc_dB''tab$''shimmer_apq3''tab$''shimmer_apq5''tab$''shimmer_apq11'
	...'tab$''mean_autocor''tab$''mean_nhr''tab$''mean_hnr''newline$'

printline Info appended as one line without header to 'save_directory$'\'user_specified_file_name$'

elsif saving = 5
	fileappend "'save_directory$'\'user_specified_file_name$'" name'tab$'method'tab$'
	...F0floor'tab$'F0ceiling'tab$'silthresh'tab$'
	...voithresh'tab$'octcost'tab$'octjmpcost'tab$'
	...voi_uvcost'tab$'maxperfact'tab$'maxampfact'tab$'total_dur'tab$'analysed_dur'tab$'
	...medianF0'tab$'meanF0'tab$'sdF0'tab$'minF0'tab$'maxF0'tab$'nPulses'tab$'nPeriods'tab$'meanPeriod'tab$'sdPeriod'tab$'
	...pctUnvoi'tab$'nvoicebreaks'tab$'pctVoicebreaks'tab$'jitter(loc)'tab$'jitter(loc,abs)'tab$'jitter(rap)'tab$'jitterppq5)'tab$'shimmer(loc)'tab$'
	...shimmer(loc,dB)'tab$'shimmer(apq3)'tab$'shimmer(apq5)'tab$'shimmer(apq11)'tab$'autocor'tab$'NHR'tab$'HNR'newline$'
	fileappend "'save_directory$'\'user_specified_file_name$'" 'name$''tab$''method$''tab$'
	...'minimum_pitch''tab$''maximum_pitch''tab$''pitch_silence_threshold''tab$'
	...'pitch_voicing_threshold''tab$''pitch_octave_cost''tab$''pitch_octave_jump_cost''tab$'
	...'pitch_voiced_unvoiced_cost''tab$'
	...'maximum_period_factor''tab$''maximum_amplitude_factor''tab$'
	...'duration_total''tab$''duration_analysed'
	...'tab$''medianPitch''tab$''meanPitch''tab$''sdPitch''tab$''minPitch''tab$'
	...'maxPitch''tab$''nPulses''tab$''nPeriods''tab$''meanPeriod''tab$''sdPeriod''tab$'
	...'pctUnvoiced''tab$''nVoicebreaks''tab$''pctVoicebreaks''tab$''jitter_loc''tab$'
	...'jitter_loc_abs''tab$''jitter_rap''tab$''jitter_ppq5''tab$''shimmer_loc''tab$'
	...'shimmer_loc_dB''tab$''shimmer_apq3''tab$''shimmer_apq5''tab$''shimmer_apq11'
	...'tab$''mean_autocor''tab$''mean_nhr''tab$''mean_hnr''newline$'

printline Info appended as one line with header to 'save_directory$'\'user_specified_file_name$'
endif

select Pitch 'name$'
Remove
select PointProcess 'name$'_'name$'
Remove
select Sound 'name$'