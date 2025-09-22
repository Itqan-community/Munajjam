import os
import google.generativeai as genai


os.environ["GOOGLE_API_KEY"] = "AIzaSyAlfJn21LIY3jlHZn94IgPQynXpW0Qg3Kc"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def remove_repetition_with_gemini(text):
    prompt = (
        "قم بإزالة التكرار في النص التالي مع الاحتفاظ بالمعنى الكامل:\n\n"
        f"{text}\n\n"
        "وأعد كتابه النص بعد ازاله اتكرار فقط , لا تكتب الخطوات في الناتج ولا تضف اي كلمات زياده او فصلات او علامات استفهام او اي علامات زياده فقط امسح التكرار مع بقاء المعنى و الترتيب "
    )
    
    
    model = genai.GenerativeModel('gemini-2.5-flash')

    try:
        # Use the correct method: generate_content()
        response = model.generate_content(prompt)
        # Access the response content
        return response.text
    except Exception as e:
        print("Error in Gemini API:", e)
        return text
st="وَمَا جَعَلْنَا أَصْحَابَ النَّارِ إِلَّا مَلَائِكَةً وَمَا جَعَلْنَا عِدَّتَهُمْ إِلَّا فِتْنَةً لِلَّذِينَ كَفَرُوا لِيَسْتَيْقِنَا الَّذِينَ أُوتُوا الْكِتَابَ لِيَسْتَيْقِنَا الَّذِينَ أُوتُوا الْكِتَابَ وَيَزْدَادَ الَّذِينَ آمَنُوا إِيمَانًا وَلَا يَرْتَابَ الَّذِينَ أُوتُوا الْكِتَابَ وَالْمُؤْمنُونَ وَلَا يَرْتَابَ الَّذِينَ أُوتُوا الْكِتَابَ وَالْمُؤْمِنُونَ وَلِيَقُولَ الَّذِينَ فِي قُلُوبِهِمْ مَرَضٌ وَالْكَافِرُونَ مَاذَا أَرَادَ اللَّهُ بِهَذَا مَثَلًا كَذَلِكَ يُضِلُّ اللَّهُ مَنْ يَشَاءُ وَيَهْدِي مَنْ يَشَاءُ وَمَا يَعْلَمُ جُنُودَ رَبِّكَ إِلَّا هُوَ وَمَا هِيَ إِلَّا ذِكْرَى لِلْبَشَر"
cl=remove_repetition_with_gemini(st)
print(cl)
